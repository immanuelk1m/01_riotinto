"""Launch one E017 direction after provision.sh uploads the bundle."""

import json
import os
import subprocess
from pathlib import Path


base = Path("/content/E017")
direction = json.loads((base / "run_config.json").read_text())["direction"]
if direction not in ("identity", "qmap52", "qmap104"):
    raise ValueError("Invalid direction")
logs = base / "logs"
logs.mkdir(exist_ok=True)
pidfile = logs / f"{direction}.pid"
if pidfile.exists() or list((base / "results" / direction).glob("*")):
    raise RuntimeError("Existing run artifacts: use a fresh session")
command = (
    f"python -u E017.py {direction} > logs/run_{direction}.out 2>&1; rc=$?; "
    f"if [ $rc -eq 0 ]; then touch logs/DONE_{direction}; else echo $rc > logs/FAILED_{direction}; fi; exit $rc"
)
process = subprocess.Popen(
    ["bash", "-c", command],
    cwd=base,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
    env={**os.environ, "OMP_NUM_THREADS": "2", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
)
pidfile.write_text(str(process.pid))
print(f"launched E017 {direction} pid={process.pid}")

