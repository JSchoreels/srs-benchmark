"""port_parity.py with the port in float64 (diagnostic)."""
import os; sys_path = os.getcwd()
import sys; sys.path.insert(0, sys_path)
import runpy, sys, torch
import models.fsrs_v7_instant as m
orig = m.FSRS7Instant.__init__
def init(self, config, w):
    orig(self, config, w)
    self.model.double(); self.m = self.m.double(); self.v = self.v.double()
    self.clipper._lo = None
m.FSRS7Instant.__init__ = init
orig_obs = m.FSRS7Instant.observe
m.FSRS7Instant.observe = lambda self, h, dt, y, off: orig_obs(self, h.double(), dt, y, off)
sys.argv = ["port_parity.py"] + sys.argv[1:]
runpy.run_path("runs/port_parity.py", run_name="__main__")
