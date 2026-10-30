"""Parity table for the upstream PR: upstream's script.py on users 1-100 (runs/parity.sh).

usage: FI_CACHE=runs/cache python runs/parity_compare.py

1. FSRS-7 (script.py) vs the published per-user FSRS-7 rows;
2. FSRS-7-Instant (script.py --instant, Python port) vs the Rust replay of quick-300 on the
   same users and fold fits: per-user LogLoss difference and per-review |p| difference;
3. FSRS-7-Instant vs FSRS-7, both from script.py, with paired bootstrap 95% CIs.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import fsrs_instant as fi  # noqa: E402
from fsrs_instant_compare import METRICS, boot_diff, load, wmean  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
PAR = HERE.parent / "srs-benchmark-parity"
NAME = "FSRS-7-short-secs-equalize_test_with_non_secs"
INST = "FSRS-7-instant-short-secs-equalize_test_with_non_secs"
base = load(PAR / "result" / f"{NAME}.jsonl")
inst = load(PAR / "result" / f"{INST}.jsonl")
pub = load(HERE / "result" / f"{NAME}.jsonl")
rust = load(HERE / "runs" / "quick-300" / "FSRS-Instant.jsonl")
users = sorted(set(base) & set(inst) & set(pub) & set(rust))
sizes = np.array([base[u]["size"] for u in users], dtype=float)
print(f"{len(users)} users, {int(sizes.sum()):,} test reviews\n")


def vals(rows, m):
    return np.array([rows[u]["metrics"][m] for u in users], dtype=float)


print("1. FSRS-7, script.py vs published (per-user mean; max |per-user difference|)\n")
print("| | " + " | ".join(METRICS) + " |\n| --- |" + " --- |" * len(METRICS))
print("| published | " + " | ".join(f"{np.nanmean(vals(pub, m)):.4f}" for m in METRICS) + " |")
print("| script.py | " + " | ".join(f"{np.nanmean(vals(base, m)):.4f}" for m in METRICS) + " |")
print("| max abs diff | " + " | ".join(
    f"{np.nanmax(np.abs(vals(base, m) - vals(pub, m))):.1e}" for m in METRICS) + " |")
print(f"\nusers whose test-review count differs from the published row: "
      f"{sum(base[u]['size'] != pub[u]['size'] for u in users)}\n")

# 2. Python port vs Rust replay
CFG = [{"name": "FSRS-Instant", "beta1": 0.9, "beta2": 0.999, "eps": 1e-08, "lr": 0.001,
        "user_bias_lr": 0.1, "again_lr": 0.01}]
dll = np.abs(vals(inst, "LogLoss") - vals(rust, "LogLoss"))
max_dp, n_cmp = 0.0, 0
for u in users:
    prep = fi._prepare(u)
    st = prep["stream"]
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
        fi._write_input(src, st, CFG)
        subprocess.run([fi.REPLAY, str(src), str(dst)], check=True)
        raw = np.fromfile(dst, dtype="<f8")
    p_rs = dict(zip(st["review_th"], raw[: len(st["review_th"])]))
    tsv = pd.read_csv(PAR / "evaluation" / INST / f"{u}.tsv", sep="\t", usecols=["review_th", "p"])
    d = np.abs(tsv["p"].to_numpy() - np.array([p_rs[r] for r in tsv["review_th"]]))
    max_dp, n_cmp = max(max_dp, float(d.max())), n_cmp + len(d)
print("2. FSRS-7-Instant, Python port (script.py --instant) vs Rust replay, same fold fits\n")
print(f"per-user LogLoss: max |diff| {dll.max():.1e}, mean {dll.mean():.1e}; "
      f"per review: max |p diff| {max_dp:.1e} over {n_cmp:,} test reviews\n")

print("3. FSRS-7-Instant vs FSRS-7, both from script.py\n")
for scale, w in (("Per user (README)", np.ones_like(sizes)), ("Review-weighted", sizes)):
    print(f"| {scale} | " + " | ".join(METRICS) + " |\n| --- |" + " --- |" * len(METRICS))
    b = {m: vals(base, m) for m in METRICS}
    print("| FSRS-7 | " + " | ".join(f"{wmean(b[m], w):.4f}" for m in METRICS) + " |")
    cells = []
    for m in METRICS:
        v = vals(inst, m)
        d, (lo, hi) = boot_diff(b[m], v, w)
        cells.append(f"{wmean(v, w):.4f} ({100 * d / wmean(b[m], w):+.1f}%, [{lo:+.4f}, {hi:+.4f}])")
    print("| FSRS-7-Instant | " + " | ".join(cells) + " |\n")
