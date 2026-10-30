"""Paired comparison of FSRS-Instant runs against FSRS-7 (fsrs_instant.py output).

usage: python fsrs_instant_compare.py <run_dir> [--baseline FSRS-7] [--users file]
       [--exclude file] [--published result/FSRS-7-...jsonl]

Means are reported per user (as in the README table) and review-weighted. Each
difference vs the baseline comes with a paired bootstrap 95% CI over users
(10,000 resamples, seed 42).
"""

import argparse
import sys
import json
from pathlib import Path

import numpy as np

METRICS = ["LogLoss", "RMSE(bins)", "AUC"]


def load(path):
    rows = {}
    for line in Path(path).open(encoding="utf-8"):
        d = json.loads(line)
        rows[d["user"]] = d
    return rows


def boot_diff(a, b, w, n=10_000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = b - a
    ww = w[idx]
    means = (d[idx] * ww).sum(1) / ww.sum(1)
    return np.average(d, weights=w), np.percentile(means, [2.5, 97.5])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--baseline", default="FSRS-7")
    ap.add_argument("--users")
    ap.add_argument("--exclude")
    ap.add_argument(
        "--published", default="result/FSRS-7-short-secs-equalize_test_with_non_secs.jsonl"
    )
    ap.add_argument("--only", help="comma-separated config names")
    args = ap.parse_args()

    run = Path(args.run_dir)
    base = load(run / f"{args.baseline}.jsonl")
    users = set(base)
    if args.users:
        users &= {int(x) for x in Path(args.users).read_text().split()}
    if args.exclude:
        users -= {int(x) for x in Path(args.exclude).read_text().split()}
    names = sorted(p.stem for p in run.glob("*.jsonl") if p.stem != args.baseline)
    if args.only:
        names = [n for n in args.only.split(",") if n in names]
    runs = {n: load(run / f"{n}.jsonl") for n in names}
    for n in names:
        users &= set(runs[n])
    pub = load(args.published) if Path(args.published).exists() else {}
    users = sorted(u for u in users if u in pub) if pub else sorted(users)
    sizes = np.array([base[u]["size"] for u in users], dtype=float)
    print(f"{len(users)} users, {int(sizes.sum()):,} test reviews\n")

    def vals(rows, m):
        return np.array([rows[u]["metrics"][m] for u in users], dtype=float)

    for scale, w in (("per user (README)", np.ones_like(sizes)), ("review-weighted", sizes)):
        print(f"## {scale}\n")
        print("| Model | " + " | ".join(METRICS) + " |")
        print("| --- |" + " --- |" * len(METRICS))
        b = {m: vals(base, m) for m in METRICS}
        if pub:
            print(
                "| FSRS-7 (published, same users) | "
                + " | ".join(f"{np.average(vals(pub, m), weights=w):.4f}" for m in METRICS)
                + " |"
            )
        print(
            f"| {args.baseline} (this run) | "
            + " | ".join(f"{np.average(b[m], weights=w):.4f}" for m in METRICS)
            + " |"
        )
        for n in names:
            cells = []
            for m in METRICS:
                v = vals(runs[n], m)
                mean = np.average(v, weights=w)
                d, (lo, hi) = boot_diff(b[m], v, w)
                rel = 100 * d / np.average(b[m], weights=w)
                cells.append(f"{mean:.4f} ({rel:+.2f}%, Δ {d:+.4f} [{lo:+.4f}, {hi:+.4f}])")
            print(f"| {n} | " + " | ".join(cells) + " |")
        print()

    for n in names:
        r = runs[n]
        secs = sum(r[u].get("replay_secs", 0) for u in users)
        reviews = sum(r[u].get("stream_reviews", 0) for u in users)
        dp = max(r[u].get("max_abs_dp_vs_base", 0) for u in users)
        if reviews:
            print(
                f"{n}: {1e6 * secs / reviews:.2f} µs per review "
                f"({reviews:,} replayed), max |p - p_FSRS-7| {dp:.2g}"
            )


if __name__ == "__main__":
    main()
