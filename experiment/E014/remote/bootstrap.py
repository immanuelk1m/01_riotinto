"""Create the remote bundle directories before Colab CLI uploads."""

from pathlib import Path


base = Path("/content/E014")
for relative in ("data", "features", "logs", "results"):
    (base / relative).mkdir(parents=True, exist_ok=True)
print(f"ready {base}")
