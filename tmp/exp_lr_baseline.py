"""Run the LR sweep with the OLD muon scaling rule (baseline arm).

muon.py on disk now carries the Moonshot rule, so this injects
tmp/muon_baseline.py (old rule) under the module name Net.Optimizers.muon
BEFORE the sweep script imports it. `from Net.Optimizers.muon import Muon`
resolves through sys.modules, so the injection wins.
"""
import importlib.util
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

spec = importlib.util.spec_from_file_location(
    "Net.Optimizers.muon", os.path.join(ROOT, "tmp", "muon_baseline.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
assert getattr(mod, "BASELINE_RULE", False), "injected module is not the baseline copy"
sys.modules["Net.Optimizers.muon"] = mod
print("injected OLD-rule muon as Net.Optimizers.muon")

sys.argv = ["exp_lr_current.py", "baseline", "1e-3,3e-3,1e-2"]
runpy.run_path(os.path.join(ROOT, "tmp", "exp_lr_current.py"), run_name="__main__")
