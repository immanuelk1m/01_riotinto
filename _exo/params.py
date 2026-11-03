"""The only place a number may enter the evaluation pipeline.

Why this file exists
--------------------
The E022–E027 audit found the round's ranking was decided by literals that no
document justified: ``_pick(min_rows=120)``, a hard-coded ``'2024-06-01'`` recency
filter, ``clip(±0.5)``, ``ffill(limit=8)``, ``pcr(k=3)``, ``ridge(lam=1.0)``,
``lasso(alpha=1e-3)`` and five gradient-boosting settings.  Those are research
degrees of freedom, and a leaderboard built on them cannot say whether a model or
a lucky constant produced the number.

So a constant may appear in the pipeline only if it is registered here with a
``source`` that says where its authority comes from:

``protocol``
    A clause of ``PROTOCOL.md`` fixes the value.  Changing it is a protocol
    revision, not a tuning step.
``definitional``
    The value is forced by the meaning of the object (a standard deviation needs
    two observations; eight horizons are the repository's registered grid).
``declared``
    A design choice that is **written down before the run** and has a stated
    rationale.  A ``declared`` constant that decides a conclusion must be shown
    not to matter by the sensitivity table in ``docs/SENSITIVITY.md``.

There is deliberately no ``tuned``/``knob``/``default`` source: an entry with an
unknown origin cannot be represented, so it cannot be imported by accident.
``tools/params_lint.py`` enforces that code outside this file does not carry
numeric literals in a position that can change a fitted object.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

ALLOWED_SOURCES = ("protocol", "definitional", "declared")

#: bump when a value changes; it is part of every leaderboard row
PARAMS_VERSION = "2.0"

#: the protocol revision these constants implement (PROTOCOL.md §0)
PROTOCOL_VERSION = "1.1"


@dataclass(frozen=True)
class Param:
    """One registered constant."""

    value: float | int | None
    source: str
    rationale: str

    def __post_init__(self) -> None:
        if self.source not in ALLOWED_SOURCES:
            raise ValueError(
                f"source {self.source!r} is not allowed; a constant of unknown origin "
                f"cannot enter the pipeline (allowed: {ALLOWED_SOURCES})"
            )
        if not self.rationale.strip():
            raise ValueError("a registered constant needs a rationale")


# --------------------------------------------------------------------- registry

REGISTRY: dict[str, Param] = {
    # -- PROTOCOL.md: the gates, quoted clause by clause --------------------
    "CAP": Param(
        1.0, "protocol",
        "PROTOCOL.md §2 Gate A: err = min(1.0, |expm1(forecast - log_return_target)|).",
    ),
    "BLOCK": Param(
        26, "protocol",
        "PROTOCOL.md §5.4: the paired interval must come from a moving-block bootstrap "
        "with 26-week blocks.",
    ),
    "SEALED_ORIGINS": Param(
        52, "protocol",
        "PROTOCOL.md §4.2: the most recent 52 weeks of mature targets are sealed.",
    ),
    "ALPHA": Param(0.05, "protocol", "PROTOCOL.md §5.5: the declared family-wise level."),
    "MIN_COVERAGE": Param(
        0.10, "protocol", "PROTOCOL.md §5.6 and §6.1 item 3: deploy coverage floor."
    ),
    "SAFETY_TOL": Param(
        0.05, "protocol",
        "PROTOCOL.md §6.1 item 4: no single series may regress by more than 0.05 MAPE.",
    ),
    # -- forced by the definition of the object ------------------------------
    "H": Param(
        8, "definitional",
        "The repository's registered horizon grid h = 1..8 weeks (PROTOCOL.md §5.1 cell "
        "definition); a different grid is a different experiment, not a tuning knob.",
    ),
    "Z_MIN_OBS": Param(
        2, "definitional",
        "A standard deviation is undefined below two observations, so an expanding "
        "z-score cannot start earlier.",
    ),
    "PCT": Param(100.0, "definitional", "Percent conversion of a fraction."),
    "SEED": Param(
        0, "definitional",
        "Bootstrap and estimator seed; fixed so a re-run reproduces the cells instead of "
        "being a second draw.",
    ),
    # -- declared before the run, sensitivity-checked -----------------------
    "LAGS": Param(
        4, "declared",
        "One month of weekly lags of the target log-return (and of the exogenous "
        "changes). Swing-tested over {2,4,8} in docs/SENSITIVITY.md.",
    ),
    "TRAIN_FRAC": Param(
        0.70, "declared",
        "Warm-up fraction of the walk-forward: the first 70 % of the usable origin grid "
        "is in-sample, the rest is walk-forward. Declared in the round's README before "
        "the run.",
    ),
    "REFIT_EVERY": Param(
        4, "declared",
        "Refit cadence in weeks. Weekly refits add no information at weekly resolution "
        "and multiply the compute by four.",
    ),
    "MIN_TRAIN": Param(
        100, "declared",
        "Minimum training window in weeks before the first forecast is issued.",
    ),
    "MIN_WEEKS": Param(
        200, "declared",
        "A target panel needs at least 200 weekly observations (≈ 4 years) to be in the "
        "study at all.",
    ),
    "MAX_WEEKS": Param(
        1040, "declared",
        "History is truncated at 20 years so that a panel from 1970 and one from 2005 are "
        "scored on comparable sample sizes.",
    ),
    "TOL_ABS": Param(
        1e-6, "definitional",
        "Float comparison tolerance when a tool re-derives a stored mean; it decides "
        "whether arithmetic reproduces, not what is estimated.",
    ),
    "TOL_REL": Param(
        1e-9, "definitional",
        "Relative twin of TOL_ABS.",
    ),
    "BOOT": Param(
        1000, "declared",
        "Bootstrap draws; 1000 resolves the 2.5 % quantile used by the interval.",
    ),
}

#: ``P["CAP"]`` and ``P.CAP`` both work; values are plain numbers
class _Access(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - programming error
            raise AttributeError(name) from exc


P = _Access({name: param.value for name, param in REGISTRY.items()})


def canonical() -> str:
    """Stable JSON of the registry, used for the hash and for provenance."""
    return json.dumps(
        {name: {"value": p.value, "source": p.source} for name, p in sorted(REGISTRY.items())},
        sort_keys=True,
        separators=(",", ":"),
    )


#: 16-hex fingerprint written into every leaderboard row this pipeline produces
PARAMS_HASH = hashlib.sha256(canonical().encode()).hexdigest()[:16]


def by_source(source: str) -> dict[str, Param]:
    return {name: param for name, param in REGISTRY.items() if param.source == source}


def describe() -> str:
    """Human table of the registry, used by the docs build."""
    lines = [
        "| constant | value | source | rationale |",
        "|---|---|---|---|",
    ]
    for name, param in sorted(REGISTRY.items(), key=lambda kv: (kv[1].source, kv[0])):
        lines.append(f"| `{name}` | `{param.value}` | {param.source} | {param.rationale} |")
    lines += ["", f"`PARAMS_VERSION = {PARAMS_VERSION}`, `PARAMS_HASH = {PARAMS_HASH}`"]
    return "\n".join(lines)


__all__ = [
    "ALLOWED_SOURCES",
    "P",
    "PARAMS_HASH",
    "PARAMS_VERSION",
    "PROTOCOL_VERSION",
    "REGISTRY",
    "Param",
    "by_source",
    "canonical",
    "describe",
]
