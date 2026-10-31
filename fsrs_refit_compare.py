"""Table for fsrs_refit.py runs: each row vs FSRS-7 with fsrs-rs per-fold fits.

usage: python fsrs_refit_compare.py <refit_dir> <rs_dir> <quick_dir>
Per-user means and paired user-bootstrap 95% CIs on the LogLoss difference.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
refit, rs, quick = (Path(p) for p in sys.argv[1:4])


def load(p):
    return {json.loads(line)["user"]: json.loads(line) for line in open(p, encoding="utf-8")}


base = load(rs / "FSRS-7.jsonl")
rows = {"FSRS-7, per-fold fits (fsrs-rs)": base}
for every, label in ((0, "daily"), (100, "every 100 reviews"), (10, "every 10 reviews"), (1, "every review")):
    for mode in ("latest", "scheduling"):
        p = refit / f"refit-{every}-{mode}.jsonl"
        if p.exists():
            rows[f"re-fit {label}, {'newest fit' if mode == 'latest' else 'scheduling time'}"] = load(p)
rows["FSRS7-Instant, at review time"] = load(rs / "FSRS7-Instant.jsonl")
rows["FSRS7-Instant, scheduling time (benchmark fits)"] = load(quick / "curve-FSRS7-Instant.jsonl")
users = sorted(set(base).intersection(*rows.values()))
assert all(r[u]["size"] == base[u]["size"] for r in rows.values() for u in users)
rng = np.random.default_rng(0)
b = np.array([base[u]["metrics"]["LogLoss"] for u in users])
print(f"{len(users)} users, {sum(base[u]['size'] for u in users):,} test reviews\n")
print("| Per user | LogLoss | vs per-fold fits | RMSE(bins) | AUC | fits per user | fit CPU-s per user |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for name, r in rows.items():
    v = np.array([r[u]["metrics"]["LogLoss"] for u in users])
    d = v - b
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    ci = "" if r is base else f"{100 * d.mean() / b.mean():+.1f}%, [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]"
    rmse = np.mean([r[u]["metrics"]["RMSE(bins)"] for u in users])
    auc = np.mean([r[u]["metrics"]["AUC"] for u in users])
    fits = np.mean([r[u]["fits"] for u in users]) if "fits" in r[users[0]] else 5
    secs = np.mean([r[u]["fit_secs"] for u in users]) if "fits" in r[users[0]] else float("nan")
    print(f"| {name} | {v.mean():.4f} | {ci} | {rmse:.4f} | {auc:.4f} | {fits:.0f} | {secs:.1f} |")
