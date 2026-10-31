"""Start one sequential GPU process in a fresh E010 workspace."""
import os
import subprocess
from pathlib import Path

base = Path("/content/E010")
logs = base / "logs"
logs.mkdir(parents=True, exist_ok=True)
pidfile = logs / "runner.pid"
commodities = ("gold", "copper", "corn", "coffee", "wti_crude_oil")
expected = {f"predictions_{c}.parquet" for c in commodities}
names = {p.name for p in (base / "results" / "gru_reg").glob("predictions_*.parquet")}
curves_complete = all((logs / f"{c}_gru_reg_curves.json").exists() for c in commodities)
running = False
if pidfile.exists():
    try:
        pid = int(pidfile.read_text())
        running = "E010.py" in Path(f"/proc/{pid}/cmdline").read_text()
    except (ValueError, FileNotFoundError, ProcessLookupError):
        pass
if running:
    print(f"E010 already running pid={pid}")
elif (logs / "DONE_gru_reg").exists() and names == expected and curves_complete:
    print("E010 already completed")
elif pidfile.exists() or names or (logs / "DONE_gru_reg").exists() or (logs / "FAILED_gru_reg").exists():
    raise RuntimeError("E010 incomplete/stale run: inspect artifacts and use a fresh workspace")
else:
    command = "python -u E010.py gru_reg > logs/run_gru_reg.out 2>&1; rc=$?; if [ $rc -eq 0 ]; then touch logs/DONE_gru_reg; else echo $rc > logs/FAILED_gru_reg; fi; exit $rc"
    process = subprocess.Popen(["bash", "-c", command], cwd=base, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, start_new_session=True,
                               env={**os.environ, "OMP_NUM_THREADS": "2", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    pidfile.write_text(str(process.pid))
    print(f"launched E010 pid={process.pid}")
