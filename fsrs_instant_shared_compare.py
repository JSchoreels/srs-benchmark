"""Protocol B table: FSRS-Instant from one shared fit vs published rows, on RWKV's test users.

usage: python fsrs_instant_shared_compare.py <run_dir>

Every row is restricted to the same users (those in the run and in every published file)
and compared with the published FSRS-7 row (per-user refit, the README's FSRS-7) by a
paired bootstrap 95% CI over users (10,000 resamples, seed 42), per user and
review-weighted.
"""

import sys
from pathlib import Path

import numpy as np

from fsrs_instant_compare import METRICS, boot_diff, load, wmean

PUBLISHED = [
    ("FSRS-7 (published, per-user refit)", "FSRS-7-short-secs-equalize_test_with_non_secs"),
    ("FSRS-7 recency (published)", "FSRS-7-short-secs-recency-equalize_test_with_non_secs"),
    ("GRU (published)", "GRU-short-secs-equalize_test_with_non_secs"),
    ("RWKV-Curve (published)", "RWKV"),
    ("RWKV-Instant (published)", "RWKV-P"),
]
OURS = [
    ("FSRS-7 shared fit, no updates", "FSRS-7-shared"),
    ("FSRS-Instant, instant", "FSRS-Instant"),
    ("FSRS-Instant, scheduling time", "FSRS-Instant-scheduling"),
    ("FSRS-Instant, start of day", "FSRS-Instant-start-of-day"),
    ("parameters only lr 3e-4, instant", "params-only-lr3e-4"),
    ("parameters only lr 3e-4, scheduling time", "params-only-lr3e-4-scheduling"),
    ("parameters only lr 3e-4, start of day", "params-only-lr3e-4-start-of-day"),
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    run = Path(sys.argv[1])
    rows = []
    for label, name in PUBLISHED:
        path = Path("result") / f"{name}.jsonl"
        if path.exists():
            rows.append((label, load(path)))
    for label, name in OURS:
        path = run / f"{name}.jsonl"
        if path.exists():
            rows.append((label, load(path)))
    users = set.intersection(*(set(r) for _, r in rows))
    users = sorted(users)
    base = rows[0][1]
    sizes = np.array([base[u]["size"] for u in users], dtype=float)
    mismatch = [
        (label, sum(r[u]["size"] != base[u]["size"] for u in users)) for label, r in rows
    ]
    print(
        f"{len(users)} users (ids {users[0]}-{users[-1]}), {int(sizes.sum()):,} test reviews; "
        "users whose test-review count differs from FSRS-7's: "
        + ", ".join(f"{label} {k}" for label, k in mismatch if k)
        + ("none" if not any(k for _, k in mismatch) else "")
        + "\n"
    )

    def vals(r, m):
        return np.array([r[u]["metrics"][m] for u in users], dtype=float)

    b = {m: vals(base, m) for m in METRICS}
    for scale, w in (("Per user (README)", np.ones_like(sizes)), ("Review-weighted", sizes)):
        print(f"| {scale} | " + " | ".join(f"{m}{'↑' if m == 'AUC' else '↓'}" for m in METRICS) + " |")
        print("| --- |" + " --- |" * len(METRICS))
        for i, (label, r) in enumerate(rows):
            cells = []
            for m in METRICS:
                v = vals(r, m)
                mean = wmean(v, w)
                if i == 0:
                    cells.append(f"{mean:.4f}")
                    continue
                d, (lo, hi) = boot_diff(b[m], v, w)
                rel = 100 * d / wmean(b[m], w)
                cells.append(f"{mean:.4f} ({rel:+.1f}%, [{lo:+.4f}, {hi:+.4f}])")
            print(f"| {label} | " + " | ".join(cells) + " |")
        print()

    for label, name in OURS:
        path = run / f"{name}.jsonl"
        if not path.exists() or name.endswith(("-scheduling", "-start-of-day")):
            continue
        r = load(path)
        secs = sum(r[u]["replay_secs"] for u in users)
        reviews = sum(r[u]["stream_reviews"] for u in users)
        load_secs = sum(r[u]["load_secs"] for u in users)
        print(
            f"{label}: {1e6 * secs / reviews:.2f} µs per replayed review, all three modes "
            f"({reviews:,} replayed); data loading {1e6 * load_secs / reviews:.0f} µs per review"
        )


if __name__ == "__main__":
    main()
