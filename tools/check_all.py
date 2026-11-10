#!/usr/bin/env python3
"""Run every mechanical gate in one command.

The purge is only as good as the checks that keep it from regressing. Each gate
below fails for a different reason, and all of them must pass:

| gate | fails when |
|---|---|
| `params_inventory --check` | a literal has no classification |
| `params_lint` | a knob-carrying estimator, an unregistered constant, a literal in a knob position, or a hard-coded data filter appears in the code |
| `leaderboard_lint` | an eligible row lacks provenance, a metric is not a MAPE, or the migrated table lost a row |
| `leaderboard_report --check` | `LEADERBOARD_*.md` disagrees with the database |
| `pytest _exo/test_leaderboard.py` | the ledger tables stop round-tripping through the in-memory schema, or the TSCV origins change |
| `evaluator_reimpl --check` | the engine's cell APE or a statement MAPE disagrees with an evaluator written from PROTOCOL.md alone |

Usage::

    uv run python tools/check_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

GATES = (
    ("inventory", [PY, "tools/params_inventory.py", "--check"]),
    ("params", [PY, "tools/params_lint.py"]),
    ("leaderboard", [PY, "tools/leaderboard_lint.py"]),
    ("report", [PY, "tools/leaderboard_report.py", "--check"]),
    ("ledger", [PY, "-m", "pytest", "_exo", "-q"]),
    ("evaluator", [PY, "tools/evaluator_reimpl.py", "--check"]),
)


def main() -> int:
    results = []
    for name, command in GATES:
        done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        tail = (done.stdout or done.stderr).strip().splitlines()
        results.append((name, done.returncode, tail[-1] if tail else ""))
        print(f"[{'PASS' if done.returncode == 0 else 'FAIL'}] {name}: {results[-1][2]}")
    failed = [name for name, code, _ in results if code != 0]
    print(f"\n[check_all] {len(results) - len(failed)}/{len(results)} gates pass"
          + (f" — failing: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
