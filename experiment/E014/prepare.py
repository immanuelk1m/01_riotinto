"""Build E014 sheet-defined feature snapshots from PostgreSQL.

Approved target/feature pairs reuse the release-safe values already materialized
in ``ml.<panel>``.  Sheet variables without an approved ``ml.feature`` row are
included from ``public.value`` at their recorded ``dt`` with zero extra lag, as
explicitly requested for this diagnostic experiment.  The manifest marks those
columns so the resulting scores cannot be mistaken for point-in-time evidence.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import psycopg
from psycopg import sql


EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
FEATURE_DIR = EXP_DIR / "features"
SPREADSHEET_ID = "17mSdLi4RUOboC-ycC_QB4m0JhBBRedbU"
DSN = "dbname=commodity"

TABS = {
    "wti_crude_oil": {"title": "WTI", "gid": "1713512547", "expected": 31},
    "copper": {"title": "구리", "gid": "1321443879", "expected": 30},
    "gold": {"title": "금", "gid": "194458146", "expected": 24},
    "corn": {"title": "옥수수", "gid": "955092015", "expected": 35},
    "wheat": {"title": "밀", "gid": "1217593986", "expected": 30},
    "coffee": {"title": "커피", "gid": "526411853", "expected": 20},
}

TARGETS = {
    "gold": "Com_LBMA_Gold_Price_PM_USD_V_0",
    "copper": "Com_LME_Cu_Cash_Bid",
    "corn": "Com_Barchart_Cash_National_Corn_Index",
    "coffee": "Com_CEPEA_ESALQ_Daily_Coffee_Robusta_Spot_USD_Per_60kg_Bag",
    "wti_crude_oil": "Oil_EIA_Cushing_WTI_Spot_Price_Daily_USD_Per_Barrel",
    "wheat": "Com_Barchart_Cash_Soft_Red_Wheat_Index",
}


def fetch_tab(gid: str) -> tuple[list[str], str]:
    url = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/export?format=csv&gid={gid}"
    payload = urlopen(url, timeout=30).read()
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    if not rows:
        raise ValueError(f"Empty sheet tab gid={gid}")
    cover_col = next((name for name in rows[0] if "커버 시계열" in name), None)
    coverage_col = next((name for name in rows[0] if name.strip() == "커버리지"), None)
    if not cover_col or not coverage_col:
        raise ValueError(f"Required columns missing from gid={gid}")
    pool = sorted({
        row[cover_col].strip()
        for row in rows
        if row.get(coverage_col, "").strip() == "O" and row.get(cover_col, "").strip()
    })
    return pool, hashlib.sha256(payload).hexdigest()


def read_panel(cur: psycopg.Cursor, panel: str) -> pd.DataFrame:
    cur.execute(sql.SQL("SELECT * FROM ml.{} ORDER BY week_end").format(sql.Identifier(panel)))
    frame = pd.DataFrame(cur.fetchall(), columns=[item.name for item in cur.description])
    frame["week_end"] = pd.to_datetime(frame["week_end"])
    return frame.set_index("week_end").astype(float)


def approved_columns(cur: psycopg.Cursor, target: str, pool: list[str]) -> dict[str, dict]:
    cur.execute(
        """SELECT f.col_name, f.effective_lag_weeks, f.policy_version,
                  f.evidence_class, c.sql_ident
             FROM ml.feature f
             JOIN ml.column_map c
               ON c.target_id=f.target_id
              AND c.feature=f.col_name || '_lag_' || f.effective_lag_weeks::text
            WHERE f.target_id=%s AND f.col_name=ANY(%s)""",
        (target, pool),
    )
    return {
        col_name: {
            "effective_lag_weeks": int(lag),
            "policy_version": policy,
            "evidence_class": evidence,
            "source_column": source_column,
        }
        for col_name, lag, policy, evidence, source_column in cur.fetchall()
    }


def raw_weekly(cur: psycopg.Cursor, names: list[str], index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    if not names:
        return {}
    cur.execute(
        """SELECT DISTINCT ON (col_name, dt) col_name, dt, value
             FROM public.value
            WHERE col_name=ANY(%s)
            ORDER BY col_name, dt, updated_at DESC, created_at DESC""",
        (names,),
    )
    raw = pd.DataFrame(cur.fetchall(), columns=["col_name", "dt", "value"])
    found = set(raw.col_name.unique()) if not raw.empty else set()
    if missing := set(names) - found:
        raise ValueError(f"Missing public.value series: {sorted(missing)}")
    result = {}
    for name, group in raw.groupby("col_name", sort=False):
        series = group.assign(dt=pd.to_datetime(group.dt)).set_index("dt").value.astype(float)
        result[name] = series.resample("W-SUN").last().reindex(index)
    return result


def read_price(cur: psycopg.Cursor, col_name: str) -> pd.DataFrame:
    cur.execute(
        """SELECT DISTINCT ON (dt) dt, value
             FROM public.value WHERE col_name=%s
            ORDER BY dt, updated_at DESC, created_at DESC""",
        (col_name,),
    )
    price = pd.DataFrame(cur.fetchall(), columns=["dt", "value"])
    if price.empty:
        raise ValueError(f"Missing target price {col_name}")
    return price


def build() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "E014_sheet_pool",
        "spreadsheet_id": SPREADSHEET_ID,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "timing_policy": {
            "approved": "reuse ml panel values with effective_lag_weeks",
            "unapproved": "public.value dt resampled W-SUN with zero extra lag",
            "warning": "Unapproved timing can leak publication or revision information.",
        },
        "tabs": {},
    }
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        for commodity, tab in TABS.items():
            pool, sheet_hash = fetch_tab(tab["gid"])
            if len(pool) != tab["expected"]:
                raise ValueError(f"{commodity}: expected {tab['expected']} sheet variables, got {len(pool)}")
            panel = read_panel(cur, commodity)
            approved = approved_columns(cur, commodity, pool)
            forced = sorted(set(pool) - set(approved))
            forced_data = raw_weekly(cur, forced, panel.index)
            output = pd.DataFrame(index=panel.index)
            variables = []
            for name in pool:
                if name in approved:
                    metadata = approved[name]
                    source_column = metadata["source_column"]
                    if source_column not in panel:
                        raise ValueError(f"{commodity}: approved column missing from panel: {source_column}")
                    output[name] = panel[source_column]
                    variables.append({
                        "col_name": name,
                        "timing_status": "approved",
                        "unapproved_timing": False,
                        **metadata,
                    })
                else:
                    output[name] = forced_data[name]
                    variables.append({
                        "col_name": name,
                        "timing_status": "forced_dt_lag0",
                        "unapproved_timing": True,
                        "effective_lag_weeks": 0,
                        "policy_version": None,
                        "evidence_class": None,
                        "source_column": "public.value",
                    })
            if list(output.columns) != pool or output.columns.duplicated().any():
                raise AssertionError(f"{commodity}: feature pool mismatch")
            coverage = output.notna().mean()
            for variable in variables:
                variable["non_null_fraction"] = float(coverage[variable["col_name"]])
                variable["first_valid_week"] = (
                    str(output[variable["col_name"]].first_valid_index().date())
                    if output[variable["col_name"]].first_valid_index() is not None else None
                )
                variable["last_valid_week"] = (
                    str(output[variable["col_name"]].last_valid_index().date())
                    if output[variable["col_name"]].last_valid_index() is not None else None
                )
            output.rename_axis("week_end").reset_index().to_parquet(DATA_DIR / f"{commodity}_X.parquet", index=False)
            read_price(cur, TARGETS[commodity]).to_parquet(DATA_DIR / f"{commodity}_price.parquet", index=False)
            (FEATURE_DIR / f"{commodity}_features.json").write_text(
                json.dumps({"features": [{"name": name} for name in pool]}, indent=2) + "\n"
            )
            manifest["tabs"][commodity] = {
                "title": tab["title"],
                "gid": tab["gid"],
                "csv_sha256": sheet_hash,
                "feature_count": len(pool),
                "approved_count": len(approved),
                "forced_count": len(forced),
                "variables": variables,
            }
            print(f"{commodity}: features={len(pool)} approved={len(approved)} forced={len(forced)}")
    (FEATURE_DIR / "sheet_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


if __name__ == "__main__":
    build()
