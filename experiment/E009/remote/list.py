from pathlib import Path

base = Path("/content/E009")
for folder in (base / "results" / "gru", base / "logs"):
    for path in sorted(folder.glob("*")):
        if path.is_file():
            print(path)
