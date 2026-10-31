"""FSRS-7 fitted by fsrs-rs's own trainer (`fsrs::benchmark`, FSRS-7).

Runs the fit-v7 tool (JSchoreels/fsrs-rs, branch fsrs-instant, tools/fit-v7) on one
training set. fsrs-rs weights training items by 0.0667 + 0.9333*(i/N)^11.25, the same as
the benchmark's `--recency`. FR_FIT is the path to fit-v7.exe.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def rs_fit(train_set):
    """fsrs-rs FSRS-7 parameters for one training set, and the fit seconds."""
    tr = train_set.sort_values("review_th")
    items = [
        np.vstack([h.numpy(), [[d, r]]])
        for h, d, r in zip(tr["tensor"], tr["delta_t"], tr["rating"])
    ]
    offsets = np.concatenate([[0], np.cumsum([len(x) for x in items])]).astype("<i8")
    reviews = np.vstack(items)
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
        with open(src, "wb") as f:
            f.write(np.array([len(items), len(reviews)], "<i4").tobytes())
            f.write(offsets.tobytes())
            f.write(reviews[:, 0].astype("<f4").tobytes())
            f.write(reviews[:, 1].astype("<i4").tobytes())
            f.write(tr["card_id"].to_numpy().astype("<i8").tobytes())
        env = {**os.environ, "RAYON_NUM_THREADS": "1"}
        subprocess.run([os.environ["FR_FIT"], str(src), str(dst)], check=True, env=env)
        raw = np.fromfile(dst, dtype=np.uint8)
    w = np.frombuffer(raw[:136].tobytes(), "<f4").astype(float).round(6).tolist()
    return w, float(np.frombuffer(raw[136:].tobytes(), "<f8")[0])
