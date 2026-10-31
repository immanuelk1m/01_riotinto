import os
import subprocess
from pathlib import Path

base = Path("/content/E011")
logs = base / "logs"
logs.mkdir(parents=True, exist_ok=True)
if (logs / "runner.pid").exists() or list((base / "results").glob("**/predictions_*.parquet")):
    raise RuntimeError("Use a fresh E011 workspace; prior run artifacts exist")
command = "python -u E011.py delay > logs/run_delay.out 2>&1; rc=$?; if [ $rc -eq 0 ]; then touch logs/DONE_delay; else echo $rc > logs/FAILED_delay; fi; exit $rc"
process = subprocess.Popen(["bash", "-c", command], cwd=base, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, start_new_session=True,
                           env={**os.environ, "OMP_NUM_THREADS": "2"})
(logs / "runner.pid").write_text(str(process.pid))
print(f"launched E011 pid={process.pid}")
