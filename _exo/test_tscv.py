"""The TSCV origins: first Sunday-ending week of each month, 2025-01 .. 2026-09."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _exo import engine  # noqa: E402


def test_tscv_origins():
    index = pd.date_range("2020-01-05", "2026-11-01", freq="W-SUN")
    got = index[engine.tscv_origins(index)]
    assert len(got) == 21
    assert got[0] == pd.Timestamp("2025-01-05") and got[-1] == pd.Timestamp("2026-09-06")
    assert (got.day_of_week == 6).all() and got.to_period("M").is_unique
    # a missing week drops its origin instead of shifting it to the next week
    assert len(engine.tscv_origins(index.drop(pd.Timestamp("2025-03-02")))) == 20
