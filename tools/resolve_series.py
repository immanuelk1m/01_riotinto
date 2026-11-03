#!/usr/bin/env python3
"""Write the resolved series into ``config.json`` and prune directions to the retained models.

The v1 engine resolved a regex per key at prepare time, with a recency filter and a
``(rows, last_dt)`` tie-break.  That made *which physical series represents "corn"* a
runtime decision — the largest research degree of freedom in the round.  The
resolution was recorded in ``data/manifest.json``, so the choice is recoverable; this
tool writes it down explicitly and from then on the config is the record.

It also removes directions whose model is not retained. What was removed is listed
under ``retired_directions`` so the round's design stays readable.

Usage::

    uv run python tools/resolve_series.py E023 E024            # dry run
    uv run python tools/resolve_series.py E023 E024 --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _exo.engine import RETAINED_MODELS  # noqa: E402

#: direction renames: the old name claimed a model that never ran
RENAMES = {"arimax_rain": "adl_rain"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="+")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    for experiment in args.experiments:
        base = ROOT / "experiment" / experiment
        config_path = base / "config.json"
        manifest_path = base / "data" / "manifest.json"
        if not config_path.exists() or not manifest_path.exists():
            print(f"[resolve] {experiment}: missing config.json or data/manifest.json, skipped")
            continue
        cfg = json.loads(config_path.read_text())
        manifest = json.loads(manifest_path.read_text())

        unavailable: dict[str, str] = cfg.get("unavailable", {})
        for kind in ("targets", "exog"):
            resolved = {}
            for key, pattern in cfg[kind].items():
                entry = pattern if isinstance(pattern, dict) else {"pattern": pattern}
                record = manifest.get(kind, {}).get(key)
                if not record:
                    # v1 dropped this key silently at prepare time, which shrank a panel without
                    # saying so. v2 records it and removes it from the directions explicitly.
                    unavailable[f"{kind}:{key}"] = (
                        "no series matched in the v1 manifest; the panel never contained it"
                    )
                    print(f"[resolve] {experiment} {kind}:{key}: unavailable, removed from the design")
                    continue
                entry["col_name"] = record["col_name"]
                entry.setdefault("name_en", record.get("name_en"))
                resolved[key] = entry
            cfg[kind] = resolved
        cfg["unavailable"] = unavailable

        kept, retired = [], cfg.get("retired_directions", [])
        for direction in cfg["directions"]:
            missing_targets = [k for k in direction["targets"] if k not in cfg["targets"]]
            if missing_targets:
                print(f"[resolve] {experiment}: {direction['name']} drops unavailable targets "
                      f"{missing_targets}")
                direction["targets"] = [k for k in direction["targets"] if k in cfg["targets"]]
                # an exog key may also be declared as a target (E023 uses the copper 3-month
                # futures price as a basis regressor), so both namespaces count as available
                available = set(cfg["targets"]) | set(cfg["exog"])
                direction["exog"] = [k for k in direction["exog"] if k in available]
            if direction["model"] in RETAINED_MODELS:
                name = direction["name"]
                if name in RENAMES:
                    print(f"[resolve] {experiment}: renaming {name} -> {RENAMES[name]} (it was plain OLS)")
                    direction["name"] = RENAMES[name]
                kept.append(direction)
            else:
                retired.append(dict(name=direction["name"], model=direction["model"],
                                    reason="model carries free parameters; see experiment/RETIRED.md"))
                print(f"[resolve] {experiment}: dropping {direction['name']} ({direction['model']})")
        cfg["directions"] = kept
        cfg["retired_directions"] = retired
        cfg["params_version"] = "2.0"

        text = json.dumps(cfg, indent=2) + "\n"
        if args.apply:
            shutil.copy(config_path, config_path.with_suffix(".json.v1"))
            config_path.write_text(text, encoding="utf-8")
            print(f"[resolve] {experiment}: wrote {len(kept)} retained directions, "
                  f"{len(retired)} retired")
        else:
            print(f"[resolve] {experiment}: dry run, would keep {len(kept)} and retire {len(retired)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
