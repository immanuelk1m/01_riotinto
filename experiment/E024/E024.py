"""E0xx entry shim: `python E0xx.py prepare`, `python E0xx.py <direction>`."""
import pathlib
import sys

BASE = pathlib.Path(__file__).resolve().parent
for candidate in (BASE, BASE.parent, BASE.parent.parent):
    sys.path.insert(0, str(candidate))

from _exo import engine  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: E0xx.py prepare | <direction> [--limit=N]")
    if sys.argv[1] == "prepare":
        engine.prepare(BASE)
    else:
        raise SystemExit(engine.run(BASE, " ".join(sys.argv[1:])))
