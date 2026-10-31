"""Paired table for fsrs_curve.py runs: each variant vs FSRS-7 fitted on the same data.

usage: python fsrs_curve_compare.py <run_dir>

Per-user (README) and review-weighted means with paired user-bootstrap 95% CIs on the
difference, the same on existing cards only (cards still in the cards table), and the
fit time relative to FSRS-7.
"""

import sys
from pathlib import Path

import numpy as np

from fsrs_instant_compare import METRICS, boot_diff, load, wmean

sys.stdout.reconfigure(encoding="utf-8")
run = Path(sys.argv[1])
base = load(run / "FSRS-7.jsonl")
names = sorted(p.stem for p in run.glob("*.jsonl") if p.stem != "FSRS-7")
rows = {n: load(run / f"{n}.jsonl") for n in names}
users = sorted(set(base).intersection(*rows.values()))
print(f"{len(users)} users, {sum(base[u]['size'] for u in users):,} test reviews\n")


def table(title, get, size):
    w_user = np.array([1.0 if size(base[u]) else np.nan for u in users])
    w_rev = np.array([float(size(base[u]) or 0) for u in users])
    for scale, w in (("Per user", w_user), ("Review-weighted", w_rev)):
        ok = ~np.isnan(w) & (w > 0)
        print(f"| {title}, {scale.lower()} | " + " | ".join(METRICS) + " |")
        print("| --- |" + " --- |" * len(METRICS))
        b = {m: np.array([get(base[u], m) for u in users])[ok] for m in METRICS}
        print("| FSRS-7 | " + " | ".join(f"{wmean(b[m], w[ok]):.4f}" for m in METRICS) + " |")
        for n in names:
            cells = []
            for m in METRICS:
                v = np.array([get(rows[n][u], m) for u in users])[ok]
                d, (lo, hi) = boot_diff(b[m], v, w[ok])
                cells.append(
                    f"{wmean(v, w[ok]):.4f} ({100 * d / wmean(b[m], w[ok]):+.1f}%, [{lo:+.4f}, {hi:+.4f}])"
                )
            print(f"| {n} | " + " | ".join(cells) + " |")
        print()


def all_cards(r, m):
    return r["metrics"][m]


def existing(r, m):
    return r["existing"]["metrics"][m] if r["existing"] else np.nan


table("All test reviews", all_cards, lambda r: r["size"])
table("Existing cards only", existing, lambda r: r["existing"] and r["existing"]["size"])

fit = sum(base[u]["fit_secs"] for u in users)
print(f"FSRS-7 fit: {fit / 3600:.2f} CPU-h ({fit / len(users):.1f} s per user)")
for n in names:
    t = sum(rows[n][u]["fit_secs"] for u in users)
    print(f"{n} fit: {t / 3600:.2f} CPU-h ({t / fit:.2f}x FSRS-7)")
