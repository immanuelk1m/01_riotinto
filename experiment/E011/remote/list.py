from pathlib import Path

base = Path("/content/E011")
for folder in (base / "results" / "delay", base / "logs"):
    for path in sorted(folder.glob("*")):
        if path.is_file() and path.name != "runner.pid":
            print(path)
