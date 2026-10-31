"""Report remote E014 direction status and the training-log tail."""

import json
from pathlib import Path


base = Path("/content/E014")
direction = json.loads((base / "run_config.json").read_text())["direction"]
commodities = ("gold", "copper", "corn", "coffee", "wti_crude_oil", "wheat")
logs = base / "logs"
names = {path.name for path in (base / "results" / direction).glob("predictions_*.parquet")}
expected = {f"predictions_{commodity}.parquet" for commodity in commodities}
complete = names == expected and all(
    (logs / f"{commodity}_{direction}_{suffix}").exists()
    for commodity in commodities
    for suffix in ("curves.json", "transfer.csv", "summary.json")
)
running = False
try:
    pid = int((logs / f"{direction}.pid").read_text())
    running = "E014.py" in Path(f"/proc/{pid}/cmdline").read_text()
except (ValueError, FileNotFoundError, ProcessLookupError):
    pass
done = (logs / f"DONE_{direction}").exists()
failed_marker = (logs / f"FAILED_{direction}").exists()
finished = done and complete and not failed_marker
failed = failed_marker or (done and not complete) or (not running and not finished)
print(f"STATUS {direction} done={len(names & expected)}/6 running={running} finished={finished} failed={failed}")
out = logs / f"run_{direction}.out"
if out.exists():
    print("\n".join(out.read_text().splitlines()[-8:]))
