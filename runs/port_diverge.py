import os, sys, subprocess, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, os.getcwd()); ARGS = sys.argv[1:]
import fsrs_instant as fi
from models.fsrs_v7_instant import instant_predict
CFG = [{"name": "x", "beta1": 0.9, "beta2": 0.999, "eps": 1e-08, "lr": 0.001, "user_bias_lr": 0.1, "again_lr": 0.01}]
cfg0 = [{"name": "x", "beta1": 0.9, "beta2": 0.999, "eps": 1e-08, "lr": 0.0}]
import models.fsrs_v7_instant as m
if os.environ.get("ROUND32"):
    _orig = m.FSRS7Instant._adam_step
    def _step(self, grad):
        _orig(self, grad)
        self.w[:] = [float(np.float32(x)) for x in self.w]
    m.FSRS7Instant._adam_step = _step
user = int(ARGS[0]); config = fi.config
if len(ARGS) > 1: m.LR = 0.0; m.USER_BIAS_LR = 0.0; m.AGAIN_LR = 0.0; CFG = cfg0
dataset = fi.UserDataLoader(config).load_user_data(user)
prep = fi._prepare(user); st = prep["stream"]
testsets = [dataset[dataset["review_th"] >= dataset["review_th"].max() + 1] for _ in range(0)]
# predict EVERY stream row: one test set per fold covering all stream rows of that fold
fs = [int(t["review_th"].min()) for t in [dataset[dataset[f"{k}_test"]] for k in range(5)]]
rows = dataset[dataset["review_th"] >= fs[0]]
fold = np.searchsorted(fs, rows["review_th"].to_numpy(), side="right") - 1
testsets = [rows[fold == k] for k in range(5)]
out = instant_predict(dataset, prep["w_list"], testsets, config)
p_of = {r: p for t in out for r, p in zip(t["review_th"], t["p"])}
with tempfile.TemporaryDirectory() as tmp:
    src, dst = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
    fi._write_input(src, st, CFG); subprocess.run([fi.REPLAY, str(src), str(dst)], check=True)
    raw = np.fromfile(dst, dtype="<f8")
n = len(st["review_th"]); p_rs = raw[:n]
p_py = np.array([p_of[r] for r in st["review_th"]])
d = np.abs(p_py - p_rs)
print("max", d.max(), "first >1e-5 at stream index", int(np.argmax(d > 1e-5)) if (d > 1e-5).any() else None, "of", n)
i = int(np.argmax(d > 1e-5)) if (d > 1e-5).any() else 0
for j in range(max(0, i - 3), min(n, i + 3)):
    c = st["card"][j]; h = st["hist_t"][st["offsets"][c]:st["offsets"][c] + st["prefix"][j]]; hr = st["hist_r"][st["offsets"][c]:st["offsets"][c] + st["prefix"][j]]
    print(j, "fold", st["fold"][j], "py", p_py[j], "rs", p_rs[j], "dt", st["delta_t"][j], "hist", list(zip(np.round(h, 4), hr)))
