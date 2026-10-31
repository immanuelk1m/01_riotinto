from pathlib import Path

base = Path("/content/E009")
logs = base / "logs"
expected = {f"predictions_{commodity}.parquet" for commodity in
            ("gold", "copper", "corn", "coffee", "wti_crude_oil")}
outputs = sorted((base / "results" / "gru").glob("predictions_*.parquet"))
names = {path.name for path in outputs}
done_marker = (logs / "DONE_gru").exists()
failed_marker = (logs / "FAILED_gru").exists()
running = False
pidfile = logs / "runner.pid"
if pidfile.exists():
    try:
        pid = int(pidfile.read_text())
        running = "E009.py" in Path(f"/proc/{pid}/cmdline").read_text()
    except (ValueError, FileNotFoundError, ProcessLookupError):
        pass
finished = done_marker and expected.issubset(names)
failed = failed_marker or (done_marker and not finished) or (pidfile.exists() and not running and not finished)
print(f"STATUS gru done={len(expected & names)}/5 running={running} finished={finished} failed={failed}")
out = logs / "run_gru.out"
if out.exists():
    print("\n".join(out.read_text().splitlines()[-8:]))
if failed:
    reason = ((logs / "FAILED_gru").read_text().strip() if failed_marker
              else "runner stopped or outputs incomplete")
    print("FAILED reason=" + reason)
