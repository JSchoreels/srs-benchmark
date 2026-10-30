"""FSRS-Instant from one shared FSRS-7 fit, on RWKV's train/test users (protocol B).

srs-benchmark's RWKV is trained on users 5000-10000 and tested on users 1-4999
(rwkv/train_rwkv_config.toml, rwkv/get_result_config.toml). Here:

  fit     fits ONE FSRS-7 parameter set on a sample of the training users, with the
          benchmark's own FSRS-7 trainer (script._fit_trainable_weights) on all their
          reviews pooled;
  replay  replays each test user's whole review stream from that shared set through
          FSRS-Instant (no per-user fit) and scores the published test reviews (the
          FSRS-7 --equalize_test_with_non_secs test folds, which are also RWKV's), in
          three modes: instant, scheduling time and start of day.

Options are environment variables (worker processes re-import this module):
  FI_OUT        output directory: shared.json from fit; <config>[-<mode>].jsonl from replay
  FI_USERS      replay: comma-separated user ids, or a file with one id per line
  FI_TRAIN      fit: the same, for the training users
  FI_CONFIGS    replay: JSON list of configs, as in fsrs_instant.py ("curve" is ignored:
                every config is scored in all three modes)
  FI_PROCESSES  worker processes (default 8)
  FI_REPLAY, FI_DATA as in fsrs_instant.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

COMMAND = sys.argv[1] if __name__ == "__main__" else None
import fsrs_instant as fi  # noqa: E402  (sets the FSRS-7 benchmark flags)

script, config = fi.script, fi.config
MODES = ["instant", "scheduling", "start-of-day"]


def _ids(spec):
    if os.path.exists(spec):
        return [int(x) for x in Path(spec).read_text().split()]
    return [int(x) for x in spec.split(",") if x]


def _load(user_id):
    try:
        df = fi.UserDataLoader(config).load_user_data(user_id)
    except Exception as e:  # noqa: BLE001  (too little data, etc.)
        return user_id, None, repr(e)
    df = df.drop(columns=[c for c in df.columns if c.endswith(("_train", "_test"))])
    # numpy, not torch: pickling one torch tensor per row back to the parent is very slow
    df["tensor"] = [t.numpy() for t in df["tensor"]]
    return user_id, df, None


def fit(out_dir):
    users = _ids(os.environ["FI_TRAIN"])
    t0 = time.perf_counter()
    frames, skipped = [], []
    with ProcessPoolExecutor(int(os.environ.get("FI_PROCESSES", "8"))) as ex:
        for user_id, df, err in ex.map(_load, users):
            if df is None:
                skipped.append([user_id, err])
            else:
                frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)
    pooled["tensor"] = [torch.from_numpy(t) for t in pooled["tensor"]]
    load_secs = time.perf_counter() - t0
    print(f"loaded {len(frames)} users, {len(pooled)} reviews in {load_secs:.0f} s", flush=True)
    t1 = time.perf_counter()
    weights = script._fit_trainable_weights(script._apply_recency_weighting(pooled))
    fit_secs = time.perf_counter() - t1
    params = fi.create_model(config, weights).w.detach().cpu().numpy().astype(float)
    out = {
        "users": [u for u in users if u not in {s[0] for s in skipped}],
        "skipped": skipped,
        "reviews": len(pooled),
        "load_secs": load_secs,
        "fit_secs": fit_secs,
        "parameters": [round(float(x), 6) for x in params],
    }
    (out_dir / "shared.json").write_text(json.dumps(out, indent=1))
    print(f"fit in {fit_secs:.0f} s: {out['parameters']}", flush=True)


def _shared_params():
    return np.array(
        [json.loads((Path(os.environ["FI_OUT"]) / "shared.json").read_text())["parameters"]],
        dtype=np.float32,
    )


def process_user(user_id):
    configs = fi._configs()
    t0 = time.perf_counter()
    dataset = fi.UserDataLoader(config).load_user_data(user_id)
    raw_day = pd.read_parquet(
        Path(config.data_path) / "revlogs" / f"user_id={user_id}", columns=["day_offset"]
    )["day_offset"].to_numpy()
    test_mask = np.zeros(len(dataset), dtype=bool)
    for split_i in range(config.n_splits):
        test_mask |= dataset[f"{split_i}_test"].to_numpy()
    keep = ["review_th", "card_id", "delta_t", "elapsed_days", "i", "rmse_bins_lapse", "y"]
    test_df = dataset[test_mask][[c for c in keep if c in dataset.columns]].copy()

    # The whole stream from the user's first review, one fold, the shared parameters.
    params = _shared_params()
    first = int(dataset["review_th"].min())
    st = fi._stream(user_id, dataset, [None], [dataset[dataset["review_th"] == first]], params)
    st["day"] = raw_day[st["review_th"] - 1]
    load_secs = time.perf_counter() - t0

    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
        fi._write_input(src, st, configs)
        subprocess.run([fi.REPLAY, str(src), str(dst)], check=True)
        raw = np.fromfile(dst, dtype="<f8")
    n = len(st["review_th"])
    preds = raw[: len(configs) * 3 * n].reshape(len(configs), 3, n)
    secs = raw[len(configs) * 3 * n :]
    pos = {r: i for i, r in enumerate(st["review_th"])}
    idx = np.array([pos[r] for r in test_df["review_th"]])
    w_list = [params[0].tolist()]
    out = {}
    for c, modes, s in zip(configs, preds, secs):
        for mode, pr in zip(MODES, modes):
            name = c["name"] if mode == "instant" else f"{c['name']}-{mode}"
            if c.get("lr", 0) == 0 and not any(c.get(k, 0) for k in fi.CONFIG_KEYS[5:9]):
                if mode != "instant":
                    continue  # no updates: all modes are the same
            stats = fi._stats(user_id, test_df, pr[idx], w_list)
            stats["replay_secs"] = float(s)
            stats["load_secs"] = load_secs
            stats["stream_reviews"] = int(n)
            out[name] = stats
    return user_id, out


def replay(out_dir):
    done = set()
    marker = out_dir / "done.jsonl"
    if marker.exists():
        done = {json.loads(line)["user"] for line in marker.open()}
    users = [u for u in _ids(os.environ["FI_USERS"]) if u not in done]
    with ProcessPoolExecutor(int(os.environ.get("FI_PROCESSES", "8"))) as ex:
        futures = [ex.submit(process_user, u) for u in users]
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                user_id, out = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"error: {e!r}", flush=True)
                continue
            for name, stats in out.items():
                with open(out_dir / f"{name}.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(stats) + "\n")
            # written last, so a user in done.jsonl has all its rows
            with open(marker, "a", encoding="utf-8") as f:
                f.write(json.dumps({"user": user_id}) + "\n")
            print(f"{k}/{len(users)} user {user_id}", flush=True)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    try:  # leave the CPU to other jobs first; workers inherit the priority class
        import psutil

        psutil.Process().nice(getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 10))
    except Exception:  # noqa: BLE001
        pass
    out_dir = Path(os.environ["FI_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    {"fit": fit, "replay": replay}[COMMAND](out_dir)
