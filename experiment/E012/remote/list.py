from pathlib import Path

base = Path("/content/E012")
for folder in (base / "results" / "hedge", base / "logs"):
    for path in sorted(folder.glob("*")):
        if path.is_file() and path.name != "runner.pid":
            print(path)
