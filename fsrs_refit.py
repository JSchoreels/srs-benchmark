"""FSRS-7 re-fitted from scratch during the review stream, scored in the benchmark protocol.

The benchmark fits FSRS-7 once per fold. Here fsrs-rs's FSRS-7 trainer (fsrs7_rs.py
weighting, via tools/refit-replay) re-fits it on all reviews so far, from the first test
review on: before each day's first review (FR_EVERY=0) or every FR_EVERY reviews. Each
test review is predicted twice:

  latest     with the newest fit before it;
  scheduling with the newest fit whose training reviews all come at or before the card's
             previous review, which is what a scheduler had when it set the interval.

usage: FR_OUT=<dir> FR_USERS=<file> FR_EVERY=<n> [FR_PROCESSES=8] FR_REFIT=<exe>
       FR_DATA=<dataset> python fsrs_refit.py
Writes <dir>/refit-<n>-latest.jsonl and <dir>/refit-<n>-scheduling.jsonl.
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

sys.argv = [
    "script.py",
    "--algo",
    "FSRS-7",
    "--short",
    "--secs",
    "--equalize_test_with_non_secs",
    "--data",
    os.environ.get("FR_DATA", "../anki-revlogs-10k"),
]
import pandas as pd  # noqa: E402
import script  # noqa: E402  (parses the benchmark flags above at import)
from data_loader import UserDataLoader  # noqa: E402
from utils import evaluate  # noqa: E402

config = script.config


def _input(user_id, dataset, every):
    rows = dataset.sort_values("review_th")
    test = np.zeros(len(rows), dtype=bool)
    for split_i in range(config.n_splits):
        test |= rows[f"{split_i}_test"].to_numpy()
    start = int(np.argmax(test))
    raw = pd.read_parquet(
        Path(config.data_path) / "revlogs" / f"user_id={user_id}",
        columns=["card_id", "day_offset"],
    )
    first_th = pd.Series(np.arange(1, len(raw) + 1)).groupby(raw["card_id"].to_numpy()).min()
    by_card = rows.sort_values(["card_id", "i"])
    prev = by_card.groupby("card_id")["review_th"].shift(1)
    prev = prev.fillna(by_card["card_id"].map(first_th)).astype(np.int64)
    prev_th = prev.reindex(rows.index).to_numpy()
    day = raw["day_offset"].to_numpy()[rows["review_th"].to_numpy() - 1]
    items = [
        np.vstack([h.numpy(), [[d, r]]])
        for h, d, r in zip(rows["tensor"], rows["delta_t"], rows["rating"])
    ]
    offsets = np.concatenate([[0], np.cumsum([len(x) for x in items])]).astype("<i8")
    reviews = np.vstack(items)
    parts = [
        np.array([len(items), len(reviews), start, every], "<i4"),
        offsets,
        reviews[:, 0].astype("<f4"),
        reviews[:, 1].astype("<i4"),
        rows["card_id"].to_numpy().astype("<i8"),
        rows["review_th"].to_numpy().astype("<i8"),
        prev_th.astype("<i8"),
        day.astype("<i4"),
    ]
    return rows, test, start, b"".join(p.tobytes() for p in parts)


def process_user(user_id):
    every = int(os.environ["FR_EVERY"])
    t0 = time.perf_counter()
    dataset = UserDataLoader(config).load_user_data(user_id)
    rows, test, start, blob = _input(user_id, dataset, every)
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
        src.write_bytes(blob)
        env = {**os.environ, "RAYON_NUM_THREADS": "1"}
        subprocess.run([os.environ["FR_REFIT"], str(src), str(dst)], check=True, env=env)
        raw = np.fromfile(dst, dtype="<f8")
    n = len(rows) - start
    keep = test[start:]
    df = rows.iloc[start:][keep].drop(columns=["tensor"])
    out = {}
    for mode, p in (("latest", raw[:n]), ("scheduling", raw[n : 2 * n])):
        part = df.assign(p=p[keep])
        stats, _ = evaluate(
            part["y"].tolist(), part["p"].tolist(), part, f"refit-{every}", user_id, config, []
        )
        stats.pop("parameters", None)
        stats["fits"] = int(raw[2 * n])
        stats["fit_secs"] = float(raw[2 * n + 1])
        stats["total_secs"] = time.perf_counter() - t0
        out[mode] = stats
    return user_id, out


if __name__ == "__main__":
    import multiprocessing as mp

    import psutil

    mp.set_start_method("spawn", force=True)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    every = int(os.environ["FR_EVERY"])
    out_dir = Path(os.environ["FR_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {m: out_dir / f"refit-{every}-{m}.jsonl" for m in ("latest", "scheduling")}
    done = (
        {json.loads(line)["user"] for line in files["scheduling"].open()}
        if files["scheduling"].exists()
        else set()
    )
    users = [int(x) for x in Path(os.environ["FR_USERS"]).read_text().split()]
    users = [u for u in users if u not in done]
    with ProcessPoolExecutor(int(os.environ.get("FR_PROCESSES", "8"))) as ex:
        futures = {ex.submit(process_user, u): u for u in users}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                user_id, out = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"error: user {futures[fut]}: {e!r}", flush=True)
                continue
            # scheduling last, so its file marks a user as fully written
            for m in ("latest", "scheduling"):
                with open(files[m], "a", encoding="utf-8") as f:
                    f.write(json.dumps(out[m]) + "\n")
            print(f"{k}/{len(users)} user {user_id}", flush=True)
