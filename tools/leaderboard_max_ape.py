#!/usr/bin/env python3
"""Add and backfill ``leaderboard.max_ape_pct``: the worst single TSCV cell.

``mape`` is a mean over cells and hides the tail.  ``max_ape_pct`` stores the single
worst cell of the same cell set, expressed in percent, but **without** the protocol's
1.0 cap: a capped maximum saturates at 100.0 for almost every series and carries no
information.

Two cell conventions are in the database:

* price space (E001-E020): ``ape = |pred / y_true - 1| * 100``
* log-return space (E021, E022+): ``ape = |expm1(fc - target_logret)| * 100``

Two artefact shapes are handled:

* price space (E012-E020): ``results/<direction>/predictions_<commodity>.parquet``
* log-return cells (E022-E026): ``results|colab_runs/*/results/<direction>_cells.parquet``
  scored on the development window (last ``SEALED_ORIGINS`` origins excluded)

E021 stored capped scores only, so its rows stay NULL until the experiment is re-run.

Every value is gated: the mean recomputed from the artefact must reproduce the stored
``mape`` (and ``n``) before the maximum is trusted.  A mismatch means the wrong column
or the wrong origin grid was picked, so the run aborts instead of writing.

Usage (repository root, ``uv run python tools/leaderboard_max_ape.py ...``):

    migrate            # idempotent ALTER TABLE ... ADD COLUMN max_ape_pct REAL
    backfill           # dry run: report what would be written
    backfill --apply   # write the values
    report             # per-experiment fill rate and worst-cell summary
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _exo.params import P  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLUMN = "max_ape_pct"

#: experiment-name suffix -> prediction column in the artefacts
SUFFIX_COLUMNS = (("_candidate", "y_cand"), ("_raw", "y_raw"), ("_equal", "y_equal"))
PERSISTENCE_NAMES = ("naive", "E015_naive")

#: experiments handled here; E021/E022+ populate the column when they run
DEFAULT_EXPERIMENTS = tuple(f"E{n:03d}" for n in (12, 13, 14, 15, 16, 18, 19, 20))

#: mirrors ``_exo/engine.py``; the gate still has to confirm the cut reproduces ``mape``
SEALED_ORIGINS = P.SEALED_ORIGINS
CELLS_GLOBS = (
    "experiment/{exp}/results/*_cells.parquet",
    "experiment/{exp}/colab_runs/*/results/*_cells.parquet",
)

TOL_ABS = P.TOL_ABS
TOL_REL = P.TOL_REL

CACHE: dict[str, dict[str, pd.DataFrame]] = {}


# ----------------------------------------------------------------- schema helpers

def connect(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db, timeout=60)
    con.execute("PRAGMA busy_timeout=60000")
    return con


def columns(con: sqlite3.Connection) -> list[str]:
    return [row[1] for row in con.execute("PRAGMA table_info(leaderboard)")]


def migrate(con: sqlite3.Connection) -> bool:
    """Add the column if it is missing.  Returns True when the schema changed."""
    if COLUMN in columns(con):
        return False
    con.execute(f"ALTER TABLE leaderboard ADD COLUMN {COLUMN} REAL")
    con.commit()
    return True


# --------------------------------------------------------------- artefact access

def split_experiment(name: str) -> tuple[str | None, str]:
    """``E014_sheet_pool_inv1_raw`` -> (``E014_sheet_pool_inv1``, ``y_raw``)."""
    if name in PERSISTENCE_NAMES:
        return None, "last_price"
    for suffix, column in SUFFIX_COLUMNS:
        if name.endswith(suffix):
            return name[: -len(suffix)], column
    return name, "y_pred"


def frames(directory: Path) -> dict[str, pd.DataFrame]:
    key = str(directory)
    if key not in CACHE:
        CACHE[key] = {
            path.stem[len("predictions_"):]: pd.read_parquet(path)
            for path in sorted(directory.glob("predictions_*.parquet"))
        }
    return CACHE[key]


def artefact_dirs(root: Path, prefix: str) -> list[Path]:
    base = root / "experiment" / prefix / "results"
    if not base.is_dir():
        return []
    return sorted(d for d in base.iterdir() if d.is_dir() and list(d.glob("predictions_*.parquet")))


def cells(frame: pd.DataFrame, column: str, horizon: str) -> pd.Series:
    """Raw (uncapped) APE percent of the scored cells behind one leaderboard row."""
    if horizon != "all":
        frame = frame[frame.horizon == int(horizon)]
    frame = frame.dropna(subset=["y_true"])
    if column not in frame.columns:
        raise KeyError(f"column {column} missing (have {sorted(frame.columns)})")
    return (frame[column] - frame.y_true).abs() / frame.y_true.abs() * 100.0


def metrics(frame: pd.DataFrame, column: str, horizon: str) -> tuple[float, float, float, int]:
    """(mean APE%, max APE%, RMSE, n) over the scored cells of one leaderboard row."""
    if horizon != "all":
        frame = frame[frame.horizon == int(horizon)]
    frame = frame.dropna(subset=["y_true"])
    if column not in frame.columns:
        raise KeyError(f"column {column} missing (have {sorted(frame.columns)})")
    error = frame[column] - frame.y_true
    ape = error.abs() / frame.y_true.abs() * 100.0
    return float(ape.mean()), float(ape.max()), float((error ** 2).mean() ** 0.5), int(len(ape))


def close(left: float, right: float) -> bool:
    return abs(left - right) <= max(TOL_ABS, TOL_REL * abs(right))


def matches(mean: float, rmse: float, n: int, stored_mean: float, stored_rmse: float | None, stored_n: int) -> bool:
    """The artefact must reproduce the stored mean, RMSE and cell count."""
    if n != stored_n:
        return False
    if stored_rmse is not None and not close(rmse, stored_rmse):
        return False
    return close(mean, stored_mean)


# --------------------------------------------------------------------- backfill

def group_rows(rows: list[sqlite3.Row]) -> dict[str, list[sqlite3.Row]]:
    """Group leaderboard rows by the artefact family they came from (``base`` name)."""
    families: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        base, _column = split_experiment(row["experiment"])
        families.setdefault(base or "<persistence>", []).append(row)
    return families


def pick_dir(root: Path, base: str | None, rows: list[sqlite3.Row]) -> tuple[Path | None, str]:
    """Find the results directory that reproduces every stored row of ``base``."""
    if base is None:  # persistence rows: any legacy directory may hold last_price
        candidates = [d for prefix in DEFAULT_EXPERIMENTS for d in artefact_dirs(root, prefix)]
    else:
        candidates = artefact_dirs(root, base.split("_")[0])
    chosen: list[tuple[Path, float]] = []
    for directory in candidates:
        store = frames(directory)
        try:
            worst = 0.0
            for row in rows:
                _base, column = split_experiment(row["experiment"])
                frame = store[row["commodity"]]
                mean, _peak, rmse, n = metrics(frame, column, row["horizon"])
                if not matches(mean, rmse, n, row["mape"], row["rmse"], row["n"]):
                    raise ValueError("mismatch")
                worst = max(worst, abs(mean - row["mape"]))
        except (KeyError, ValueError, TypeError):
            continue
        chosen.append((directory, worst))
    if not chosen:
        return None, "no artefact directory reproduces the stored mape"
    if len(chosen) > 1:
        # identical grids produce identical maxima; anything else is ambiguous
        peaks = {}
        for directory, _worst in chosen:
            store = frames(directory)
            values = tuple(
                round(metrics(store[row["commodity"]], split_experiment(row["experiment"])[1], row["horizon"])[1], 9)
                for row in rows
            )
            peaks[directory] = values
        if len(set(peaks.values())) > 1:
            return None, f"ambiguous ({len(chosen)} directories disagree): {[d.name for d, _ in chosen]}"
    named = [pair for pair in chosen if base and base.endswith(pair[0].name)]
    return min(named or chosen, key=lambda pair: pair[1])[0], ""


def backfill(con: sqlite3.Connection, root: Path, experiments: tuple[str, ...], apply: bool) -> int:
    migrate(con)
    con.row_factory = sqlite3.Row
    rows = [
        row for row in con.execute(
            "SELECT experiment, commodity, horizon, mape, rmse, n, target_series, max_ape_pct "
            "FROM leaderboard ORDER BY experiment, commodity, horizon"
        )
    ]
    scope = DEFAULT_EXPERIMENTS if not experiments else experiments
    price_rows = [row for row in rows if row["experiment"].split("_")[0] in scope or row["experiment"] in PERSISTENCE_NAMES]
    all_prefixes = {row["experiment"].split("_")[0] for row in rows}
    cells_prefixes = sorted(p for p in all_prefixes if p.startswith("E02") and cells_files(root, p))
    in_scope = price_rows
    families = group_rows(in_scope)

    updates: list[tuple[float, str, str, str]] = []
    skipped: list[tuple[str, str]] = []
    peaks_by_family: dict[str, tuple[float, float]] = {}
    for base, family_rows in sorted(families.items()):
        directory, why = pick_dir(root, None if base == "<persistence>" else base, family_rows)
        if directory is None:
            skipped.append((base, why))
            continue
        store = frames(directory)
        worst_peak, mean_peak = 0.0, 0.0
        for row in family_rows:
            _base, column = split_experiment(row["experiment"])
            mean, peak, _rmse, _n = metrics(store[row["commodity"]], column, row["horizon"])
            updates.append((round(peak, 9), row["experiment"], row["commodity"], row["horizon"]))
            worst_peak = max(worst_peak, peak)
            mean_peak = max(mean_peak, mean)
        peaks_by_family[base] = (mean_peak, worst_peak)
        print(f"[match] {base:<32} <- {directory.relative_to(root)}  rows={len(family_rows)} "
              f"max_ape_pct up to {worst_peak:.3f}")

    for base, why in skipped:
        print(f"[skip ] {base:<32} {why}")

    if cells_prefixes:
        backfill_cells(con, root, set(cells_prefixes), apply)

    print(f"\n[plan ] {len(updates)} rows to update, {len(skipped)} families skipped "
          f"({'dry run' if not apply else 'applying'})")
    if not apply:
        for peak, experiment, commodity, horizon in sorted(updates, reverse=True)[:5]:
            print(f"        {experiment:<32} {commodity:<16} h={horizon:<3} max_ape_pct={peak:.3f}")
        return 0

    with con:
        con.executemany(
            f"UPDATE leaderboard SET {COLUMN} = ? WHERE experiment = ? AND commodity = ? AND horizon = ?",
            updates,
        )
    print(f"[write] {len(updates)} rows updated")
    return 0


# --------------------------------------------------------- log-return cells

def cells_files(root: Path, prefix: str) -> list[Path]:
    files = []
    for pattern in CELLS_GLOBS:
        files.extend(sorted(root.glob(pattern.format(exp=prefix))))
    return files


def log_metrics(frame: pd.DataFrame, mape_col: str, raw_col: str, horizon: str, variant: str, series: str | None = None):
    """(mean capped APE%, max raw APE%, n) for one row of a ``*_cells.parquet`` file."""
    block = frame if variant in ("all", "asis") else frame[frame.origin < int(frame.origin.max()) - SEALED_ORIGINS]
    if series is not None and "series" in block.columns:
        block = block[block.series == series]
    if horizon != "all":
        block = block[block.horizon == int(horizon)]
    if block.empty or mape_col not in block.columns:
        raise KeyError("missing cells")
    if raw_col not in block.columns:
        # cells written before the engine stored the uncapped twins: rebuild them
        if not {"fc", "target_logret"} <= set(block.columns):
            raise KeyError("no raw APE available")
        error = np.expm1(block.fc - block.target_logret) if raw_col == "ape_cand_raw" else np.expm1(block.target_logret)
        block = block.assign(_raw=np.abs(error) * 100.0)
        raw_col = "_raw"
    return float(block[mape_col].mean()), float(block[raw_col].max()), int(len(block))


def pooled_cells(root: Path, prefix: str, common_origins: bool = False):
    """Dev-window cells pooled over every direction of one experiment (persistence is direction-free)."""
    frames = []
    for path in cells_files(root, prefix):
        frame = pd.read_parquet(path)
        frames.append(frame[frame.origin < int(frame.origin.max()) - SEALED_ORIGINS])
    if not frames:
        return None
    pooled = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["series", "origin", "horizon"])
    if not common_origins:
        return pooled
    # panel-level rows only cover origins that every series in the panel shares
    shared = set.intersection(*[set(block.origin.unique()) for _series, block in pooled.groupby("series")])
    return pooled[pooled.origin.isin(shared)]


def pick_cells(root: Path, experiment: str, prefix: str, rows: list[sqlite3.Row], persistence: bool):
    """Find the cells file (and window) that reproduces every stored row of a family."""
    candidates = []
    for path in cells_files(root, prefix):
        frame = pd.read_parquet(path)
        candidates.append((str(path.relative_to(root)), "dev", frame))
        candidates.append((str(path.relative_to(root)), "all", frame))
    if persistence:
        for flag in (False, True):
            pooled = pooled_cells(root, prefix, flag)
            if pooled is not None:
                candidates.append((f"{prefix}/* pooled{' + common origins' if flag else ''}", "asis", pooled))
    for label, variant, frame in candidates:
        try:
            for row in rows:
                mape_col = "mape_pers" if persistence else "mape_cand"
                raw_col = "ape_pers_raw" if persistence else "ape_cand_raw"
                mean, _peak, n = log_metrics(frame, mape_col, raw_col, row["horizon"], variant, row["commodity"])
                if not matches(mean, 0.0, n, row["mape"], None, row["n"]):
                    raise ValueError("mismatch")
        except (KeyError, ValueError, TypeError, IndexError):
            continue
        return label, variant, frame
    return None, "", None


def backfill_cells(con: sqlite3.Connection, root: Path, prefixes: set[str], apply: bool) -> int:
    """Backfill the E022+ rows from their ``*_cells.parquet`` artefacts."""
    con.row_factory = sqlite3.Row
    updates: list[tuple[float, str, str, str]] = []
    skipped: list[tuple[str, str]] = []
    for prefix in sorted(prefixes):
        rows = [row for row in con.execute(
            "SELECT experiment, commodity, horizon, mape, rmse, n, max_ape_pct FROM leaderboard "
            "WHERE experiment = ? OR experiment LIKE ? ORDER BY experiment, commodity, horizon",
            (f"{prefix}_persistence", f"{prefix}_%"),
        )]
        direction_rows: dict[str, list[sqlite3.Row]] = {}
        persistence_rows: list[sqlite3.Row] = []
        for row in rows:
            if row["experiment"] == f"{prefix}_persistence":
                persistence_rows.append(row)
            else:
                direction_rows.setdefault(row["experiment"], []).append(row)

        families = [(name, family, False) for name, family in sorted(direction_rows.items())]
        if persistence_rows:
            families.append((f"{prefix}_persistence", persistence_rows, True))
        for name, family_rows, persistence in families:
            label, variant, frame = pick_cells(root, name, prefix, family_rows, persistence)
            if label is None:
                skipped.append((name, "no cells file reproduces the stored mape"))
                continue
            mape_col = "mape_pers" if persistence else "mape_cand"
            raw_col = "ape_pers_raw" if persistence else "ape_cand_raw"
            peak = 0.0
            for row in family_rows:
                _mean, value, _n = log_metrics(frame, mape_col, raw_col, row["horizon"], variant, row["commodity"])
                updates.append((round(value, 9), row["experiment"], row["commodity"], row["horizon"]))
                peak = max(peak, value)
            print(f"[match] {name:<32} <- {label} [{variant}] rows={len(family_rows)} "
                  f"max_ape_pct up to {peak:.3f}")

    for name, why in skipped:
        print(f"[skip ] {name:<32} {why}")
    if updates and apply:
        with con:
            con.executemany(
                f"UPDATE leaderboard SET {COLUMN} = ? WHERE experiment = ? AND commodity = ? AND horizon = ?",
                updates,
            )
    print(f"[cells] {len(updates)} rows {'updated' if apply else 'to update'}, {len(skipped)} families skipped")
    return 0


# ----------------------------------------------------------------------- report

def report(con: sqlite3.Connection) -> int:
    con.row_factory = sqlite3.Row
    print(f"{'experiment':<34}{'rows':>6}{'filled':>8}{'null':>6}{'mape':>10}{'max_ape_pct':>13}")
    for row in con.execute(
        "SELECT experiment, count(*) AS rows, count(max_ape_pct) AS filled, "
        "avg(mape) AS mape, max(max_ape_pct) AS peak "
        "FROM leaderboard GROUP BY experiment ORDER BY experiment"
    ):
        print(f"{row['experiment']:<34}{row['rows']:>6}{row['filled']:>8}{row['rows'] - row['filled']:>6}"
              f"{row['mape']:>10.3f}{(row['peak'] if row['peak'] is not None else float('nan')):>13.3f}")
    total = con.execute("SELECT count(*) AS n, count(max_ape_pct) AS f FROM leaderboard").fetchone()
    print(f"\n[total] {total['f']}/{total['n']} rows carry {COLUMN}")
    inverted = con.execute(
        f"SELECT count(*) FROM leaderboard WHERE {COLUMN} IS NOT NULL AND {COLUMN} < mape"
    ).fetchone()[0]
    print(f"[check] rows with {COLUMN} < mape: {inverted} (must be 0)")
    return 0


# ------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["migrate", "backfill", "report"])
    parser.add_argument("--db", type=Path, default=ROOT / "leaderboard.sqlite")
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root holding experiment/")
    parser.add_argument("--experiments", default="", help="comma separated EXP prefixes (default E012-E020)")
    parser.add_argument("--apply", action="store_true", help="write the backfilled values")
    args = parser.parse_args(argv)

    experiments = tuple(part.strip() for part in args.experiments.split(",") if part.strip())
    if not args.db.exists():
        raise SystemExit(f"database not found: {args.db}")

    con = connect(args.db)
    try:
        if args.command == "migrate":
            changed = migrate(con)
            print(f"[schema] {COLUMN} {'added' if changed else 'already present'}")
            return 0
        if args.command == "backfill":
            return backfill(con, args.root, experiments, args.apply)
        return report(con)
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
