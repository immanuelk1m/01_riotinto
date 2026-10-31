from pathlib import Path

base = Path("/content/E011")
logs = base / "logs"
commodities = ("gold", "copper", "corn", "coffee", "wti_crude_oil")
expected = {f"predictions_{c}.parquet" for c in commodities}
names = {p.name for p in (base / "results" / "delay").glob("predictions_*.parquet")}
diagnostics = all((logs / f"{c}_delay_weights.csv").exists() for c in commodities)
running = False
try:
    pid = int((logs / "runner.pid").read_text())
    running = "E011.py" in Path(f"/proc/{pid}/cmdline").read_text()
except (ValueError, FileNotFoundError, ProcessLookupError):
    pass
finished = (logs / "DONE_delay").exists() and names == expected and diagnostics and (logs / "recipe.json").exists()
failed = (logs / "FAILED_delay").exists() or (not running and not finished)
finished = finished and not failed
print(f"STATUS delay done={len(names & expected)}/5 running={running} finished={finished} failed={failed}")
out = logs / "run_delay.out"
if out.exists():
    print("\n".join(out.read_text().splitlines()[-8:]))
