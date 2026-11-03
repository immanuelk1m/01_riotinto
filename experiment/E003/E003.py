"""E003 — retired 2026-11-03.

Removed model: LightGBM + Lasso/LassoCV knockoff screen.
Removed constants: LGBMRegressor; Lasso; LassoCV; array_split; rng.choice.

Why: two estimator families with free penalties screen the features before the fit. A model with a free parameter cannot be ranked against
persistence in a way that separates skill from a lucky constant, so it may not
produce leaderboard rows (see experiment/RETIRED.md).

The implementation is not deleted from history: it is at commit 2ca48e9,
path ``experiment/E003/``. The artefacts in ``data/`` and ``results/``
stay here, and the leaderboard rows are kept under ``status = 'retired-heuristic'``.
"""
from _exo.retired import RetiredExperiment

EXPERIMENT = "E003"
DIRECTIONS: tuple[str, ...] = ()
REMOVED_MODEL = "LightGBM + Lasso/LassoCV knockoff screen"
REASON = "two estimator families with free penalties screen the features before the fit"
COMMIT = "2ca48e9"


def main() -> None:
    raise RetiredExperiment(EXPERIMENT, REASON, COMMIT)


if __name__ == "__main__":
    main()
