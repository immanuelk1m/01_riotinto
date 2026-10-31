"""Run in the Colab kernel; one sequential GRU process uses the GPU."""
import os
import subprocess
from pathlib import Path

base = Path("/content/E009")
logs = base / "logs"
logs.mkdir(parents=True, exist_ok=True)
pidfile = logs / "runner.pid"
expected = {f"predictions_{commodity}.parquet" for commodity in
            ("gold", "copper", "corn", "coffee", "wti_crude_oil")}
results = {path.name for path in (base / "results" / "gru").glob("predictions_*.parquet")}
running = False
if pidfile.exists():
    try:
        pid = int(pidfile.read_text())
        running = "E009.py" in Path(f"/proc/{pid}/cmdline").read_text()
    except (ValueError, FileNotFoundError, ProcessLookupError):
        pass
if (logs / "DONE_gru").exists() and expected.issubset(results):
    print("E009 already completed")
elif running:
    print(f"E009 already running pid={pid}")
else:
    (logs / "DONE_gru").unlink(missing_ok=True)
    (logs / "FAILED_gru").unlink(missing_ok=True)
    command = "python -u E009.py gru > logs/run_gru.out 2>&1; rc=$?; if [ $rc -eq 0 ]; then touch logs/DONE_gru; else echo $rc > logs/FAILED_gru; fi; exit $rc"
    process = subprocess.Popen(["bash", "-c", command], cwd=base, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, start_new_session=True,
                               env={**os.environ, "OMP_NUM_THREADS": "2", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    pidfile.write_text(str(process.pid))
    print(f"launched E009 pid={process.pid}")
