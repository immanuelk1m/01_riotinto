from pathlib import Path

base = Path("/content/E012")
logs = base / "logs"
commodities = ("gold", "copper", "corn", "coffee", "wti_crude_oil")
expected = {f"predictions_{c}.parquet" for c in commodities}
names = {p.name for p in (base / "results" / "hedge").glob("predictions_*.parquet")}
diagnostics = all((logs / f"{c}_hedge_{kind}.csv").exists() for c in commodities for kind in ("weights", "transfer"))
running = False
try:
    pid = int((logs / "runner.pid").read_text())
    running = "E012.py" in Path(f"/proc/{pid}/cmdline").read_text()
except (ValueError, FileNotFoundError, ProcessLookupError):
    pass
finished = ((logs / "DONE_hedge").exists() and names == expected and diagnostics
            and not (logs / "FAILED_hedge").exists())
failed = (logs / "FAILED_hedge").exists() or (not running and not finished)
print(f"STATUS hedge done={len(expected & names)}/5 running={running} finished={finished} failed={failed}")
out = logs / "run_hedge.out"
if out.exists():
    print("\n".join(out.read_text().splitlines()[-8:]))
