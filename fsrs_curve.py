"""FSRS-7 curve candidates under the published FSRS-7 protocol (per-user, 5 folds).

Each variant is fitted and scored exactly like `script.py --algo FSRS-7 --short --secs
--equalize_test_with_non_secs` (parameters fixed over each test fold, so every prediction
is a scheduling-time prediction). A plain FSRS-7 run on the same data is the paired
baseline.

Variants (FC_VARIANTS, JSON list of {"name", ...}):
  {"distill": a}   soft targets in training: y' = (1 - a) * y + a * p_teacher, where
                   p_teacher is RWKV-Curve's prediction for that review (made at the
                   card's previous review). Rows without a teacher value keep y. Scored
                   on the 0/1 outcomes as usual.
  {"again": true}  Again count scales the stability (models/fsrs_v7_curve.py).
  {"sibling": true} a new card's initial difficulty blends in its siblings' current
                   difficulty. D_sib comes from the fold's plain FSRS-7 fit (the baseline),
                   from sibling reviews before the card's first review only. Cards missing
                   from the cards table have no siblings. fit_secs includes the baseline
                   fit, so it is the variant's whole cost.

Each row also has "existing": the metrics on test reviews of cards that are still in the
cards table. The teacher saw the deleted-card flag, so for distillation that subset is the
leak-free reading.

Options are environment variables (worker processes re-import this module):
  FC_OUT, FC_USERS (file, one id per line), FC_VARIANTS, FC_PROCESSES (default 8),
  FC_DATA (dataset), FC_TEACHER (teacher cache with users/<u>.npz)
"""

import json
import os
import sys
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
    os.environ.get("FC_DATA", "../anki-revlogs-10k"),
]
import script  # noqa: E402  (parses the benchmark flags above at import)
from data_loader import UserDataLoader  # noqa: E402
from models import model_factory  # noqa: E402
from models.fsrs_v7 import FSRS7  # noqa: E402
import torch  # noqa: E402
from models.fsrs_v7_curve import curve_model  # noqa: E402
from utils import Collection, evaluate  # noqa: E402

config = script.config


def _teacher(user_id, n_rows):
    """RWKV-Curve probability per review_th (index review_th - 1), NaN where missing."""
    z = np.load(Path(os.environ["FC_TEACHER"]) / "users" / f"{user_id}.npz")
    logit = z["targets"][:, 0].astype(np.float64)
    assert len(logit) >= n_rows, (user_id, len(logit), n_rows)
    return 1.0 / (1.0 + np.exp(-logit))


def _existing(user_id, card_ids):
    path = Path(config.data_path) / "cards" / f"user_id={user_id}"
    if not path.exists():
        return np.zeros(len(card_ids), dtype=bool)
    cards = pd.read_parquet(path, columns=["card_id"])
    return np.isin(card_ids, cards["card_id"].to_numpy())


def _sibling_d(user_id, dataset, w_list):
    """Per fold: card_id -> mean current difficulty of its siblings at its first review.

    A sibling's difficulty after its k-th review comes from the fold's FSRS-7 fit and
    counts once that review has happened, so only reviews before the card's first review
    enter it.
    """
    path = Path(config.data_path) / "cards" / f"user_id={user_id}"
    if not path.exists():
        return [{} for _ in w_list]
    cards = pd.read_parquet(path, columns=["card_id", "note_id"])
    note_of = dict(zip(cards["card_id"], cards["note_id"]))
    raw = pd.read_parquet(
        Path(config.data_path) / "revlogs" / f"user_id={user_id}", columns=["card_id"]
    )["card_id"].to_numpy()
    first_th = pd.Series(np.arange(1, len(raw) + 1)).groupby(raw).min()
    rows = dataset[["card_id", "i", "review_th"]].copy()
    rows["pos"] = np.arange(len(rows))
    rows = rows[rows["card_id"].map(lambda c: c in note_of)]
    rows["note"] = rows["card_id"].map(note_of)
    rows = rows[rows.groupby("note")["card_id"].transform("nunique") > 1]
    rows = rows.sort_values(["card_id", "i"])
    prev_th = rows.groupby("card_id")["review_th"].shift(1)
    prev_i = rows.groupby("card_id")["i"].shift(1)
    # the row with history length i has D after review i, known from review i's time
    rows["tau"] = np.where(
        rows["i"] == 1,
        rows["card_id"].map(first_th),
        np.where(prev_i == rows["i"] - 1, prev_th, np.nan),
    )
    rows = rows[~rows["tau"].isna()]
    out = []
    for w in w_list:
        model = model_factory.create_model(config, w)  # plain FSRS-7 (the baseline)
        d_all = np.array(Collection(model, config).batch_predict(dataset)[2])
        d = d_all[rows["pos"].to_numpy()]
        sib = {}
        for _, g in rows.assign(d=d).groupby("note"):
            card, tau, dv = g["card_id"].to_numpy(), g["tau"].to_numpy(), g["d"].to_numpy()
            for c in np.unique(card):
                t = first_th.get(c)
                m = (card != c) & (tau < t)
                if not m.any():
                    continue
                # each sibling's latest difficulty before t
                order = np.argsort(tau[m], kind="stable")
                last = pd.Series(dv[m][order], index=card[m][order])
                sib[c] = float(last.groupby(level=0).last().mean())
        out.append(sib)
    return out


def _with_sibling(dataset, sib):
    """History tensors with D_sib and a has-siblings flag as input columns 2 and 3."""
    df = dataset.copy()
    df["tensor"] = [
        torch.cat([t, torch.tensor([[sib.get(c, 0.0), float(c in sib)]]).expand(len(t), 2)], 1)
        for t, c in zip(df["tensor"], df["card_id"])
    ]
    return df


def _fit_and_predict(dataset, variant, teacher, sib=None):
    again, sibling = bool(variant.get("again")), bool(variant.get("sibling"))
    model_factory.MODEL_REGISTRY["FSRS-7"] = (
        curve_model(again, sibling) if again or sibling else FSRS7
    )
    a = float(variant.get("distill", 0.0))
    t0 = time.perf_counter()
    w_list, parts = [], []
    for split_i in range(config.n_splits):
        data = _with_sibling(dataset, sib[split_i]) if sibling else dataset
        train_set = data[data[f"{split_i}_train"]].copy()
        test_set = data[data[f"{split_i}_test"]].copy()
        if a > 0:
            pt = teacher[train_set["review_th"].to_numpy() - 1]
            y = train_set["y"].to_numpy(dtype=np.float64)
            train_set["y"] = np.where(np.isnan(pt), y, (1 - a) * y + a * pt)
        try:
            weights = script._fit_trainable_weights(train_set)
        except Exception as e:
            if not script._is_inadequate_training_data_error(e):
                raise
            weights = script.get_model_state(model_factory.create_model(config))
        w_list.append(weights)
        test_set["p"] = Collection(
            model_factory.create_model(config, weights), config
        ).batch_predict(test_set)[0]
        parts.append(test_set)
    fit_secs = time.perf_counter() - t0
    model_factory.MODEL_REGISTRY["FSRS-7"] = FSRS7
    return pd.concat(parts), w_list, fit_secs


def _stats(user_id, df, w_list):
    stats, _ = evaluate(
        df["y"].tolist(), df["p"].tolist(), df, "fsrs-curve", user_id, config, w_list
    )
    return stats


def process_user(user_id):
    variants = json.loads(os.environ["FC_VARIANTS"])
    dataset = UserDataLoader(config).load_user_data(user_id)
    teacher = None
    if any(v.get("distill") for v in variants):
        teacher = _teacher(user_id, int(dataset["review_th"].max()))
    out, sib = {}, None
    for v in [{"name": "FSRS-7"}] + variants:
        extra = 0.0
        if v.get("sibling") and sib is None:
            t0 = time.perf_counter()
            sib = _sibling_d(user_id, dataset, base_w)
            sib_secs = time.perf_counter() - t0
        if v.get("sibling"):
            extra = base_secs + sib_secs  # stage 1 (the baseline fit) + sibling inputs
        df, w_list, fit_secs = _fit_and_predict(dataset, v, teacher, sib)
        fit_secs += extra
        if v["name"] == "FSRS-7":
            base_w, base_secs = w_list, fit_secs
        df = df.drop(columns=["tensor"])
        stats = _stats(user_id, df, w_list)
        keep = _existing(user_id, df["card_id"].to_numpy())
        stats["existing"] = (
            {"size": int(keep.sum()), "metrics": _stats(user_id, df[keep], w_list)["metrics"]}
            if keep.sum() >= 2 and df[keep]["y"].nunique() == 2
            else None
        )
        stats["fit_secs"] = fit_secs
        out[v["name"]] = stats
    return user_id, out


if __name__ == "__main__":
    import multiprocessing as mp

    import psutil

    mp.set_start_method("spawn", force=True)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    out_dir = Path(os.environ["FC_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    base_file = out_dir / "FSRS-7.jsonl"
    done = {json.loads(line)["user"] for line in base_file.open()} if base_file.exists() else set()
    users = [int(x) for x in Path(os.environ["FC_USERS"]).read_text().split()]
    users = [u for u in users if u not in done]
    with ProcessPoolExecutor(int(os.environ.get("FC_PROCESSES", "8"))) as ex:
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
