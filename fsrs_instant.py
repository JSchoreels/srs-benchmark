"""FSRS-Instant benchmark driver: FSRS-7 re-fitted online after every review.

Runs the published FSRS-7 protocol (--algo FSRS-7 --short --secs
--equalize_test_with_non_secs) and, on the same per-fold parameters, replays each
user's review stream through the fsrs-rs FSRS-Instant model
(fsrs-rs/tools/instant-replay): every review in a test fold's time span is predicted
with the current parameters, then one Adam step is taken on its outcome. A new fold
restarts from that fold's fitted parameters, so FSRS-Instant = baseline + online updates
and the comparison is paired.

Driver options are environment variables (worker processes re-import this module):
  FI_OUT       output directory (one <config>.jsonl per config, plus FSRS-7.jsonl)
  FI_USERS     comma-separated user ids, or a file with one id per line
  FI_CONFIGS   JSON list of {"name", "lr", "beta1", "beta2", "eps", "l2", and optionally
               "user_bias_lr", "card_bias_lr", "note_bias_lr", "again_lr", "curve"}
               (curve 1: score the scheduling-time predictions instead of the instant ones)
  FI_PROCESSES worker processes (default 8)
  FI_REPLAY    path to the instant-replay executable
  FI_DATA      dataset path (default ../anki-revlogs-10k)
  FI_CACHE     optional directory caching each user's fit and stream, so other
               configs replay without re-fitting FSRS-7
"""

import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.argv = [
    "script.py",
    "--algo",
    "FSRS-7",
    "--short",
    "--secs",
    "--equalize_test_with_non_secs",
    "--data",
    os.environ.get("FI_DATA", "../anki-revlogs-10k"),
]
import script  # noqa: E402  (parses the benchmark flags above at import)
from data_loader import UserDataLoader  # noqa: E402
from models.model_factory import create_model  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
from utils import Collection, evaluate  # noqa: E402

config = script.config
REPLAY = os.environ.get(
    "FI_REPLAY",
    str(
        Path(__file__).resolve().parent.parent
        / "fsrs-rs/tools/instant-replay/target/release/instant-replay.exe"
    ),
)


def _configs():
    return json.loads(os.environ["FI_CONFIGS"])


def _fold_weights(dataset):
    """Per-fold weights and test sets, exactly as script.process for --partitions none."""
    w_list, testsets = [], []
    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    for split_i, _ in enumerate(tscv.split(dataset)):
        train_set = dataset[dataset[f"{split_i}_train"]]
        test_set = dataset[dataset[f"{split_i}_test"]]
        assert train_set["partition"].nunique() <= 1
        try:
            weights = script._fit_trainable_weights(
                script._apply_recency_weighting(train_set.copy())
            )
        except Exception as e:
            if not script._is_inadequate_training_data_error(e):
                raise
            weights = script.get_model_state(create_model(config))
        w_list.append(weights)
        testsets.append(test_set)
    return w_list, testsets


CONFIG_KEYS = ["lr", "beta1", "beta2", "eps", "l2", "user_bias_lr", "card_bias_lr",
               "note_bias_lr", "again_lr", "curve"]


def _note_index(user_id, card_ids):
    """Per-row note index (siblings share it), -1 for cards missing from the cards table.

    Cards deleted before the export have no cards row. They get no note bias at all
    rather than a note of their own, so the model never behaves differently because a
    card will be deleted later (the deleted-card leak).
    """
    path = Path(config.data_path) / "cards" / f"user_id={user_id}"
    if not path.exists():  # a few users have no cards partition at all
        return np.full(len(card_ids), -1, dtype=np.int32)
    cards = pd.read_parquet(path, columns=["card_id", "note_id"])
    note_of = dict(zip(cards["card_id"], cards["note_id"]))
    codes = {}
    return np.array(
        [codes.setdefault(note_of[c], len(codes)) if c in note_of else -1 for c in card_ids],
        dtype=np.int32,
    )


def _stream(user_id, dataset, w_list, testsets, params=None):
    """The replay input that does not depend on the configs, as arrays.

    The stream starts at the first test set; `params` (one row per test set) defaults to
    the FSRS-7 parameters of `w_list`.
    """
    fold_start = [int(t["review_th"].min()) for t in testsets]
    rows = dataset[dataset["review_th"] >= fold_start[0]].sort_values("review_th")
    fold = np.searchsorted(fold_start, rows["review_th"].to_numpy(), side="right") - 1

    # Each row's tensor is a prefix of its card's full (delta_t, rating) history; ship
    # the longest one per card and a prefix length per row.
    card_codes, card_index = np.unique(rows["card_id"].to_numpy(), return_inverse=True)
    lengths = np.array([len(t) for t in rows["tensor"]], dtype=np.int32)
    longest = {}
    for i, (c, n) in enumerate(zip(card_index, lengths)):
        if c not in longest or n > lengths[longest[c]]:
            longest[c] = i
    tensors = rows["tensor"].tolist()
    offsets = np.zeros(len(card_codes) + 1, dtype=np.int64)
    hist = []
    for c in range(len(card_codes)):
        full = tensors[longest[c]].numpy()
        hist.append(full)
        offsets[c + 1] = offsets[c] + len(full)
    for i, c in enumerate(card_index):
        full = hist[c]
        assert np.array_equal(tensors[i].numpy(), full[: lengths[i]])
    hist = np.concatenate(hist) if hist else np.zeros((0, 2), dtype=np.float32)
    if params is None:
        params = np.array(
            [create_model(config, w).w.detach().cpu().numpy() for w in w_list],
            dtype=np.float32,
        )
    return {
        "params": params,
        "offsets": offsets,
        "hist_t": hist[:, 0].astype("<f4"),
        "hist_r": hist[:, 1].astype("<i4"),
        "card": card_index.astype("<i4"),
        "prefix": lengths,
        "delta_t": rows["delta_t"].to_numpy().astype("<f4"),
        "y": rows["y"].to_numpy().astype("<i4"),
        "fold": fold.astype("<i4"),
        "note": _note_index(user_id, rows["card_id"].to_numpy()),
        "review_th": rows["review_th"].to_numpy(),
    }


def _write_input(path, st, configs):
    cfg = np.array([[c.get(k, 0.0) for k in CONFIG_KEYS] for c in configs], dtype="<f8")
    header = [len(st["params"]), len(st["offsets"]) - 1, len(st["card"]), len(configs),
              len(st["hist_t"])]
    with open(path, "wb") as f:
        np.array(header, dtype="<i4").tofile(f)
        st["params"].astype("<f4").tofile(f)
        cfg.tofile(f)
        st["offsets"].astype("<i8").tofile(f)
        for k in ("hist_t", "hist_r", "card", "prefix", "delta_t", "y", "fold", "note"):
            st[k].astype("<f4" if k in ("hist_t", "delta_t") else "<i4").tofile(f)
        # Anki day per row, for the start-of-day mode (older caches have none)
        st.get("day", np.zeros(len(st["card"]))).astype("<i4").tofile(f)


def _prepare(user_id):
    """Per-fold FSRS-7 fit, baseline predictions and replay stream (cached in FI_CACHE)."""
    cache = os.environ.get("FI_CACHE")
    path = Path(cache) / f"{user_id}.pkl" if cache else None
    if path is not None and path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    t0 = time.perf_counter()
    dataset = UserDataLoader(config).load_user_data(user_id)
    w_list, testsets = _fold_weights(dataset)
    fit_secs = time.perf_counter() - t0

    # Baseline FSRS-7: fold weights fixed over the fold, as in script.process.
    parts = []
    for w, testset in zip(w_list, testsets):
        part = testset.copy()
        part["p"] = Collection(create_model(config, w), config).batch_predict(part)[0]
        parts.append(part)
    # evaluate() needs only these; the history strings would dominate the cache size
    keep = ["review_th", "card_id", "delta_t", "elapsed_days", "i", "rmse_bins_lapse", "y", "p"]
    test_df = pd.concat(parts)[[c for c in keep if c in parts[0].columns]]
    prep = {
        "test_df": test_df,
        "w_list": w_list,
        "fit_secs": fit_secs,
        "stream": _stream(user_id, dataset, w_list, testsets),
    }
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(prep, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    return prep


def _stats(user_id, test_df, p, w_list):
    df = test_df.copy()
    df["p"] = p
    stats, _ = evaluate(
        df["y"].tolist(), df["p"].tolist(), df, "fsrs-instant", user_id, config, w_list
    )
    return stats


def process_user(user_id):
    configs = _configs()
    prep = _prepare(user_id)
    test_df, w_list, st = prep["test_df"], prep["w_list"], prep["stream"]
    p_base = test_df["p"].to_numpy()
    out = {"FSRS-7": _stats(user_id, test_df, p_base, w_list)}
    out["FSRS-7"]["fit_secs"] = prep["fit_secs"]

    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
        _write_input(src, st, configs)
        subprocess.run([REPLAY, str(src), str(dst)], check=True)
        raw = np.fromfile(dst, dtype="<f8")
    review_th = st["review_th"]
    n = len(review_th)
    # per config: instant, scheduling-time and start-of-day predictions
    modes = raw[: len(configs) * 3 * n].reshape(len(configs), 3, n)
    preds = [m[int(c.get("curve", 0))] for c, m in zip(configs, modes)]
    secs = raw[len(configs) * 3 * n :]
    pos = {r: i for i, r in enumerate(review_th)}
    idx = np.array([pos[r] for r in test_df["review_th"]])
    for c, pr, s in zip(configs, preds, secs):
        p = pr[idx]
        stats = _stats(user_id, test_df, p, w_list)
        stats["replay_secs"] = float(s)
        stats["stream_reviews"] = int(n)
        stats["max_abs_dp_vs_base"] = float(np.max(np.abs(p - p_base)))
        out[c["name"]] = stats
    return user_id, out


def _users():
    spec = os.environ["FI_USERS"]
    if os.path.exists(spec):
        return [int(x) for x in Path(spec).read_text().split()]
    return [int(x) for x in spec.split(",") if x]


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    out_dir = Path(os.environ["FI_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    base_file = out_dir / "FSRS-7.jsonl"
    if base_file.exists():
        done = {json.loads(line)["user"] for line in base_file.open()}
    users = [u for u in _users() if u not in done]
    with ProcessPoolExecutor(int(os.environ.get("FI_PROCESSES", "8"))) as ex:
        futures = [ex.submit(process_user, u) for u in users]
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                user_id, out = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"error: {e!r}", flush=True)
                continue
            # baseline last, so FSRS-7.jsonl marks a user as fully written
            for name in sorted(out, key=lambda n: n == "FSRS-7"):
                with open(out_dir / f"{name}.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(out[name]) + "\n")
            print(f"{k}/{len(users)} user {user_id}", flush=True)
