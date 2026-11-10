"""Offline check of the in-memory ledger: frames round-trip, and the primary key upserts."""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _exo import leaderboard  # noqa: E402


def test_roundtrip_and_upsert():
    empty = leaderboard.dump(leaderboard.load({}))
    row = {c: None for c in leaderboard.COLUMNS}
    row.update(experiment="E0_x", commodity="gold", horizon="1", aggregate=0, mape=1.5, n=3)
    frames = {**empty, "leaderboard": pd.DataFrame([row], columns=empty["leaderboard"].columns)}
    con = leaderboard.load(frames)
    back = leaderboard.dump(con)
    pd.testing.assert_frame_equal(back["leaderboard"], frames["leaderboard"], check_dtype=False)

    con.execute(leaderboard.INSERT, tuple({**row, "mape": 2.5}[c] for c in leaderboard.COLUMNS))
    out = leaderboard.dump(con)["leaderboard"]
    assert len(out) == 1 and out.mape.iloc[0] == 2.5
    assert isinstance(con, sqlite3.Connection)
