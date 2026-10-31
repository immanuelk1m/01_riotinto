from pathlib import Path

base = Path("/content/E010")
logs = base / "logs"
commodities = ("gold", "copper", "corn", "coffee", "wti_crude_oil")
expected = {f"predictions_{c}.parquet" for c in commodities}
names = {p.name for p in (base / "results" / "gru_reg").glob("predictions_*.parquet")}
curves_complete = all((logs / f"{c}_gru_reg_curves.json").exists() for c in commodities)
done_marker, failed_marker = (logs / "DONE_gru_reg").exists(), (logs / "FAILED_gru_reg").exists()
running = False
pidfile = logs / "runner.pid"
if pidfile.exists():
    try:
        pid = int(pidfile.read_text())
        running = "E010.py" in Path(f"/proc/{pid}/cmdline").read_text()
    except (ValueError, FileNotFoundError, ProcessLookupError):
        pass
finished = done_marker and names == expected and curves_complete and not failed_marker
failed = failed_marker or (done_marker and not finished) or (not running and not finished)
print(f"STATUS gru_reg done={len(expected & names)}/5 running={running} finished={finished} failed={failed}")
out = logs / "run_gru_reg.out"
if out.exists():
    print("\n".join(out.read_text().splitlines()[-8:]))
if failed:
    reason = (logs / "FAILED_gru_reg").read_text().strip() if failed_marker else "runner stopped or outputs incomplete"
    print("FAILED reason=" + reason)
