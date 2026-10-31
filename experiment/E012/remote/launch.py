"""Launch once in a fresh /content/E012 directory with uploaded inputs."""
import subprocess
from pathlib import Path

base = Path("/content/E012")
logs = base / "logs"
logs.mkdir(parents=True, exist_ok=True)
if (logs / "runner.pid").exists() or list((base / "results").glob("hedge/predictions_*.parquet")):
    raise RuntimeError("E012 workspace already used; inspect it before relaunch")
command = "python -u E012.py hedge > logs/run_hedge.out 2>&1; rc=$?; if [ $rc -eq 0 ]; then touch logs/DONE_hedge; else echo $rc > logs/FAILED_hedge; fi; exit $rc"
process = subprocess.Popen(["bash", "-c", command], cwd=base, start_new_session=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
(logs / "runner.pid").write_text(str(process.pid))
print(f"launched E012 pid={process.pid}")
