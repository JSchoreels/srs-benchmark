"""FSRS-7 fitted by fsrs-rs's own trainer, scored in the benchmark's protocol.

Per user and fold, the training set goes to fsrs-rs (`fsrs::benchmark`, FSRS-7, via the
fit-v7 tool in JSchoreels/fsrs-rs branch fsrs-instant), and the test fold is predicted with
the benchmark's FSRS7 model. fsrs-rs weights training items by 0.0667 + 0.9333*(i/N)^11.25,
the same as `--recency`, so the matching published row is FSRS-7-short-secs-recency-
equalize_test_with_non_secs.

usage: FR_OUT=<file.jsonl> FR_USERS=<file> [FR_PROCESSES=8] FR_FIT=<fit-v7.exe>
       FR_DATA=<dataset> python fsrs7_rs_check.py
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


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
import script  # noqa: E402  (parses the benchmark flags above at import)
import pandas as pd  # noqa: E402
from data_loader import UserDataLoader  # noqa: E402
from fsrs7_rs import rs_fit  # noqa: E402
from models import model_factory  # noqa: E402
from utils import Collection, evaluate  # noqa: E402

config = script.config


def process_user(user_id):
    t0 = time.perf_counter()
    dataset = UserDataLoader(config).load_user_data(user_id)
    load_secs = time.perf_counter() - t0
    w_list, parts, fit_secs = [], [], 0.0
    for split_i in range(config.n_splits):
        train_set = dataset[dataset[f"{split_i}_train"]]
        test_set = dataset[dataset[f"{split_i}_test"]].copy()
        w, secs = rs_fit(train_set)
        fit_secs += secs
        w_list.append(w)
        test_set["p"] = Collection(
            model_factory.create_model(config, w), config
        ).batch_predict(test_set)[0]
        parts.append(test_set)
    df = pd.concat(parts).drop(columns=["tensor"])
    stats, _ = evaluate(
        df["y"].tolist(), df["p"].tolist(), df, "FSRS-7-rs", user_id, config, w_list
    )
    stats["fit_secs"] = fit_secs
    stats["load_secs"] = load_secs
    stats["total_secs"] = time.perf_counter() - t0
    return stats


if __name__ == "__main__":
    import multiprocessing as mp

    import psutil

    mp.set_start_method("spawn", force=True)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    out = Path(os.environ["FR_OUT"])
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["user"] for line in out.open()} if out.exists() else set()
    users = [int(x) for x in Path(os.environ["FR_USERS"]).read_text().split()]
    users = [u for u in users if u not in done]
    with ProcessPoolExecutor(int(os.environ.get("FR_PROCESSES", "8"))) as ex:
        futures = {ex.submit(process_user, u): u for u in users}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                stats = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"error: user {futures[fut]}: {e!r}", flush=True)
                continue
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(stats) + "\n")
            print(f"{k}/{len(users)} user {stats['user']}", flush=True)
