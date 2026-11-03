"""E007 — retired 2026-11-03.

Removed model: LightGBM robustness variants.
Removed constants: LGBMRegressor; max_depth; n_estimators; learning_rate.

Why: each robustness variant is a different set of boosted-tree constants, so the comparison ranks constants rather than models. A model with a free parameter cannot be ranked against
persistence in a way that separates skill from a lucky constant, so it may not
produce leaderboard rows (see experiment/RETIRED.md).

The implementation is not deleted from history: it is at commit 2ca48e9,
path ``experiment/E007/``. The artefacts in ``data/`` and ``results/``
stay here, and the leaderboard rows are kept under ``status = 'retired-heuristic'``.
"""
from _exo.retired import RetiredExperiment

EXPERIMENT = "E007"
DIRECTIONS: tuple[str, ...] = ()
REMOVED_MODEL = "LightGBM robustness variants"
REASON = "each robustness variant is a different set of boosted-tree constants, so the comparison ranks constants rather than models"
COMMIT = "2ca48e9"


def main() -> None:
    raise RetiredExperiment(EXPERIMENT, REASON, COMMIT)


if __name__ == "__main__":
    main()
