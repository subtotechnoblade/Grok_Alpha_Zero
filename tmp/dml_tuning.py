"""Retest DirectML with the configuration its docs actually require.

My first benchmark used default SessionOptions, which is wrong for DML:
  - enable_mem_pattern MUST be False
  - execution_mode MUST be ORT_SEQUENTIAL
  - shapes should be known at session creation; a free batch dim degrades it,
    fixable with add_free_dimension_override_by_name
Also uses a much longer warmup, since DML compiles lazily.

Each configuration runs in its own subprocess because DirectML can hard-crash
(access violation) and would otherwise take the whole benchmark with it.

Run:
  C:\\Users\\TryHarder\\anaconda3\\envs\\tensorflow_updated\\python.exe tmp/dml_tuning.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = f"{ROOT}/tmp/dml_bench/model.onnx"
PROBE = f"{ROOT}/tmp/dml_probe_one.py"
BATCHES = [1, 4, 8, 16]
KINDS = ["cpu", "naive", "required", "fixed"]

import onnx

m = onnx.load(MODEL)
opsets = {i.domain or "ai.onnx": i.version for i in m.opset_import}
dims = [(d.dim_param or d.dim_value) for d in m.graph.input[0].type.tensor_type.shape.dim]
print(f"model      : {os.path.basename(MODEL)}")
print(f"opset      : {opsets}   (DML docs: >20 unsupported / poor performance)")
print(f"input dims : {dims}")
print()


def run(kind, batch):
    try:
        p = subprocess.run([sys.executable, PROBE, kind, str(batch)],
                           capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    for line in p.stdout.splitlines():
        if line.startswith("RESULT"):
            _, _, _, us, prov = line.split()
            return float(us), prov
    if p.returncode == -1073741819 or p.returncode == 3221225477:
        return None, "ACCESS_VIOLATION"
    err = (p.stderr or "").strip().splitlines()
    return None, (err[-1][:44] if err else f"rc={p.returncode}")


results = {}
for b in BATCHES:
    for k in KINDS:
        results[(k, b)] = run(k, b)

print(f"{'batch':>6} " + "".join(f"{k:>16}" for k in KINDS) + f"{'best vs CPU':>14}")
for b in BATCHES:
    row = f"{b:>6} "
    for k in KINDS:
        us, note = results[(k, b)]
        row += f"{(f'{us:.1f}' if us else note):>16}"
    cpu = results[("cpu", b)][0]
    dml = [results[(k, b)][0] for k in ("naive", "required", "fixed")
           if results[(k, b)][0]]
    row += f"{(f'{cpu / min(dml):.2f}x' if dml and cpu else 'n/a'):>14}"
    print(row)

print()
print("microseconds, median of 300 runs after 300 warmup runs, one process each")
print("'best vs CPU' > 1 means the best DML config beats CPU; < 1 means CPU wins")
