"""Python port (models/fsrs_v7_instant.py) vs the Rust replay, on the same fold fits.

usage: FI_CACHE=runs/cache python runs/port_parity.py <user> [<user> ...]
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
users = [int(u) for u in sys.argv[1:]]
import fsrs_instant as fi  # noqa: E402
from models.fsrs_v7_instant import instant_predict  # noqa: E402

CFG = [{"name": "FSRS-Instant", "beta1": 0.9, "beta2": 0.999, "eps": 1e-08, "lr": 0.001,
        "user_bias_lr": 0.1, "again_lr": 0.01}]
config = fi.config
for user in users:
    dataset = fi.UserDataLoader(config).load_user_data(user)
    prep = fi._prepare(user)  # cached fold fits, same as the Rust runs
    w_list, test_df = prep["w_list"], prep["test_df"]
    testsets = [dataset[dataset[f"{k}_test"]] for k in range(config.n_splits)]
    t0 = time.perf_counter()
    out = instant_predict(dataset, w_list, testsets, config)
    py_secs = time.perf_counter() - t0
    p_py = np.concatenate([t["p"].to_numpy() for t in out])
    th_py = np.concatenate([t["review_th"].to_numpy() for t in out])

    st = prep["stream"]
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
        fi._write_input(src, st, CFG)
        subprocess.run([fi.REPLAY, str(src), str(dst)], check=True)
        raw = np.fromfile(dst, dtype="<f8")
    n = len(st["review_th"])
    pos = {r: i for i, r in enumerate(st["review_th"])}
    p_rs = raw[:n][[pos[r] for r in th_py]]
    y = test_df.set_index("review_th").loc[th_py, "y"].to_numpy()

    def ll(p):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    print(f"user {user}: {n} stream reviews, python {1e6 * py_secs / n:.0f} us/review; "
          f"LogLoss python {ll(p_py):.6f} rust {ll(p_rs):.6f}; "
          f"max |dp| {np.max(np.abs(p_py - p_rs)):.2e}, mean |dp| {np.mean(np.abs(p_py - p_rs)):.2e}",
          flush=True)
