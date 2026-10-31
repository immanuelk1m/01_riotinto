from pathlib import Path

base = Path("/content/E010")
for folder in (base / "results" / "gru_reg", base / "logs"):
    for path in sorted(folder.glob("*")):
        if path.is_file() and path.name != "runner.pid":
            print(path)
