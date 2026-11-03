#!/usr/bin/env python3
"""Retire every experiment whose model carries a free parameter.

The rule is stated once, in ``experiment/RETIRED.md``, and applied uniformly: a
direction can be ranked only if its forecast comes from a model whose definition
is fixed by the protocol or by a closed form.  Everything else — LightGBM with a
genetic feature search, GRU/LSTM cores, ridge/lasso paths, boosted-tree grids, and
the post-processing wrappers that consume their forecasts — is retired.

Retirement removes the *runnable code*, not the record:

* the module becomes a stub that raises ``RetiredExperiment``,
* the artefacts (``data/``, ``results/``, ``README.md``) stay where they are,
* the implementation stays in git at the commit recorded in the stub,
* the leaderboard rows are moved to ``status = 'retired-heuristic'``.

Usage::

    uv run python tools/retire_experiment.py            # dry run: what would change
    uv run python tools/retire_experiment.py --apply    # write stubs, remove code
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiment"
RETIRED_MD = EXPERIMENTS / "RETIRED.md"
TEST = EXPERIMENTS / "test_retired.py"

#: experiment -> what was removed and why
RETIREMENTS: dict[str, dict] = {
    "E001": dict(
        model="LightGBM + genetic feature search",
        knobs="LGBMRegressor; array_split; rng.choice; GENERATIONS; PRUNE_FRAC; N_KEEP",
        reason="the forecast comes from boosted trees whose depth, learning rate and tree "
               "count were hand-set, on top of a stochastic feature search",
    ),
    "E002": dict(
        model="LightGBM with a top-k feature screen",
        knobs="LGBMRegressor; TOP_K = 30",
        reason="TOP_K decides which features enter the model and was never chosen by a rule",
    ),
    "E003": dict(
        model="LightGBM + Lasso/LassoCV knockoff screen",
        knobs="LGBMRegressor; Lasso; LassoCV; array_split; rng.choice",
        reason="two estimator families with free penalties screen the features before the fit",
    ),
    "E004": dict(
        model="LightGBM, direct/recursive/mimo variants",
        knobs="LGBMRegressor; TOP_K = 30",
        reason="every variant inherits the top-k screen and the boosted-tree settings",
    ),
    "E005": dict(
        model="Ridge over a GridSearchCV penalty grid",
        knobs="Ridge; GridSearchCV; alpha grid",
        reason="the penalty is selected by a grid search, so the reported number is the "
               "best point of a searched family",
    ),
    "E006": dict(
        model="LightGBM ensembles with four weighting schemes",
        knobs="LGBMRegressor; n_estimators = 300/200; learning_rate = 0.05",
        reason="the boosting settings and the ensemble weights are hand-set constants",
    ),
    "E007": dict(
        model="LightGBM robustness variants",
        knobs="LGBMRegressor; max_depth; n_estimators; learning_rate",
        reason="each robustness variant is a different set of boosted-tree constants, so the "
               "comparison ranks constants rather than models",
    ),
    "E008": dict(
        model="LightGBM loss/averaging variants",
        knobs="LGBMRegressor; huber/dro/global variants",
        reason="the ranking across variants is decided by the loss and averaging constants",
    ),
    "E009": dict(
        model="GRU (torch) candidate",
        knobs="GRU; hidden size; learning rate; epochs; patience",
        reason="the architecture and the training schedule are free parameters",
    ),
    "E010": dict(
        model="GRU (torch) with registration",
        knobs="GRU; hidden size; learning rate; epochs; patience",
        reason="same GRU core as E009 with a different regularisation setting",
    ),
    "E011": dict(
        model="delayed EWLS correction of E010 GRU forecasts",
        knobs="EWLS decay and memory-horizon constants on top of E010",
        reason="it post-processes GRU forecasts, so it inherits E010's free parameters and "
               "adds its own decay constant",
    ),
    "E012": dict(
        model="delayed fixed-share aggregation of E009/E010 forecasts",
        knobs="fixed-share learning rate and delay constants",
        reason="the aggregation rate is a hand-set constant applied to knob-carrying inputs",
    ),
    "E013": dict(
        model="invariant-risk GRU core",
        knobs="GRU; environment split; penalty weight",
        reason="the invariance penalty and the network are both free parameters",
    ),
    "E014": dict(
        model="sheet-pool wrapper around the E013 GRU core",
        knobs="E013 constants; pooling weights",
        reason="it wraps the E013 GRU core and adds pooling weights",
    ),
    "E015": dict(
        model="GRU pools with fresh/frozen/warm starts",
        knobs="GRU; start-up schedule",
        reason="the start-up schedule and the network are free parameters",
    ),
    "E016": dict(
        model="affine calibration of E014 forecasts",
        knobs="recent-26 and static-52 calibration windows on top of E014",
        reason="it calibrates E014's GRU output and chooses the calibration window by hand",
    ),
    "E017": dict(
        model="sheet-pool wrapper around the E013 GRU core",
        knobs="E013 constants; pooling weights",
        reason="same wrapper as E014 over the same knob-carrying core",
    ),
    "E018": dict(
        model="e-process with shrunk cross-sectional weights",
        knobs="ALPHA = 0.05; MIN_LABELS = 13; SHRINK_MIN = 13; SHRINK_FRACTION = 0.5; "
              "EFFECT_SCALE = 0.01",
        reason="five hand-set constants define the rule, including a shrinkage schedule",
    ),
    "E019": dict(
        model="adaptive-conformal (ACI) deployment rule",
        knobs="ACI_GAMMA = 0.10; LAMBDA_MAX = 0.5; MIN_LABELS = 13; RECENT = 26",
        reason="the step size of the adaptive level is a hand-set constant; it is this "
               "rule that held the program's best legacy number (6.160)",
    ),
    "E020": dict(
        model="GRU with the E014 snapshot",
        knobs="GRU; snapshot constants",
        reason="it runs the GRU core against a fixed snapshot",
    ),
    "E021": dict(
        model="multi-output ridge over 730 series",
        knobs="RIDGE_LAMBDA = 1.0; LAGS = 4; MIN_WEEKS = 250",
        reason="the ridge penalty is a constant, and this single experiment owns 53 % of "
               "the leaderboard rows",
    ),
}


def stub(experiment: str, info: dict, commit: str) -> str:
    return f'''"""{experiment} — retired 2026-11-03.

Removed model: {info['model']}.
Removed constants: {info['knobs']}.

Why: {info['reason']}. A model with a free parameter cannot be ranked against
persistence in a way that separates skill from a lucky constant, so it may not
produce leaderboard rows (see experiment/RETIRED.md).

The implementation is not deleted from history: it is at commit {commit},
path ``experiment/{experiment}/``. The artefacts in ``data/`` and ``results/``
stay here, and the leaderboard rows are kept under ``status = 'retired-heuristic'``.
"""
from _exo.retired import RetiredExperiment

EXPERIMENT = "{experiment}"
DIRECTIONS: tuple[str, ...] = ()
REMOVED_MODEL = "{info['model']}"
REASON = "{info['reason']}"
COMMIT = "{commit}"


def main() -> None:
    raise RetiredExperiment(EXPERIMENT, REASON, COMMIT)


if __name__ == "__main__":
    main()
'''


def retire_markdown() -> str:
    lines = [
        "# Retired experiments (2026-11-03)",
        "",
        "## The rule",
        "",
        "> A direction can be ranked against persistence only if its forecast comes from a",
        "> model whose definition is fixed by `PROTOCOL.md` or by a **closed form with no free",
        "> parameter**. Everything else is retired.",
        "",
        "This is an *a priori* rule about model definitions, applied to every experiment at",
        "once, without reference to how well each scored. It removes the round's nominal",
        "winner (`E024 gbrt_agri`), its nominal runner-up (`E024 combo_agri`), its worst",
        "(`E026 mlp_broad`) and the program's best legacy number (`E019 cap`, 6.160) in the",
        "same pass. That symmetry is the evidence that the rule is not results-driven.",
        "",
        "A constant is allowed to survive only if it is registered in `_exo/params.py` with",
        "a `source` (`protocol` / `definitional` / `declared`) **before** a run, and a",
        "`declared` one has to be shown not to matter by `docs/SENSITIVITY.md`. Nothing",
        "already observed can be rescued retroactively, which is why no legacy experiment",
        "survives.",
        "",
        "## What was retired",
        "",
        "| experiment | removed model | removed constants | reason |",
        "|---|---|---|---|",
    ]
    for experiment, info in sorted(RETIREMENTS.items()):
        lines.append(
            f"| `{experiment}` | {info['model']} | {info['knobs']} | {info['reason']} |"
        )
    lines += [
        "",
        "## What survives",
        "",
        "| model | why it is allowed |",
        "|---|---|",
        "| `persistence` | forecasts a zero log-return; no parameter |",
        "| `adl` | closed-form OLS on a registered design matrix; `np.linalg.lstsq`, no penalty |",
        "",
        "The leaderboard rows that survive are the persistence baselines plus the ADL",
        "directions of E023 and E024 re-run under `PARAMS_VERSION = 2.0`.",
        "",
    ]
    return "\n".join(lines)


def experiment_files(experiment: str) -> list[Path]:
    """Every .py file inside one experiment directory (main module excluded)."""
    directory = EXPERIMENTS / experiment
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.rglob("*.py"))


def apply_one(experiment: str, info: dict, commit: str, dry: bool) -> list[str]:
    directory = (EXPERIMENTS / experiment).resolve()
    if EXPERIMENTS.resolve() not in directory.parents:  # never touch anything outside experiment/
        raise SystemExit(f"refusing to modify {directory}")
    removed: list[str] = []
    main = directory / f"{experiment}.py"
    for path in experiment_files(experiment):
        if path.name == f"{experiment}.py":
            continue
        removed.append(str(path.relative_to(ROOT)))
        if not dry:
            subprocess.run(["git", "rm", "-q", str(path.relative_to(ROOT))], cwd=ROOT, check=True)
    if not dry:
        main.write_text(stub(experiment, info, commit), encoding="utf-8")
        readme = directory / "README.md"
        tombstone = (
            f"\n\n## Retired 2026-11-03\n\n"
            f"This experiment cannot run. Model: {info['model']}. "
            f"Removed constants: {info['knobs']}. Reason: {info['reason']}. "
            f"Implementation preserved at commit `{commit}`; rows kept in the leaderboard "
            f"under `status = 'retired-heuristic'`. See [`../RETIRED.md`](../RETIRED.md).\n"
        )
        if readme.exists():
            text = readme.read_text(encoding="utf-8")
            if "## Retired 2026-11-03" not in text:
                readme.write_text(text.rstrip() + tombstone, encoding="utf-8")
        else:
            readme.write_text(f"# {experiment}\n" + tombstone, encoding="utf-8")
    return removed


def test_module() -> str:
    cases = "\n".join(f'    "{e}",' for e in sorted(RETIREMENTS))
    return f'''"""A retired experiment must refuse to run; the purge is otherwise cosmetic."""
import importlib

import pytest

from _exo.retired import RetiredExperiment

RETIRED = (
{cases}
)


@pytest.mark.parametrize("experiment", RETIRED)
def test_retired_experiment_refuses_to_run(experiment):
    module = importlib.import_module(f"experiment.{{experiment}}.{{experiment}}")
    assert module.DIRECTIONS == ()
    with pytest.raises(RetiredExperiment):
        module.main()
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    dry = not args.apply

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()

    total = 0
    for experiment, info in sorted(RETIREMENTS.items()):
        if not (EXPERIMENTS / experiment).is_dir():
            print(f"[retire] {experiment}: no directory, skipped")
            continue
        removed = apply_one(experiment, info, commit, dry)
        total += len(removed)
        print(f"[retire] {experiment}: stub + {len(removed)} file(s) {'would be ' if dry else ''}removed")
    if not dry:
        RETIRED_MD.write_text(retire_markdown(), encoding="utf-8")
        TEST.write_text(test_module(), encoding="utf-8")
    print(f"[retire] {len(RETIREMENTS)} experiments, {total} files {'(dry run)' if dry else 'removed'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
