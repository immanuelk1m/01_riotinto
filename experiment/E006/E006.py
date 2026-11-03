"""E006 — retired 2026-11-03.

Removed model: LightGBM ensembles with four weighting schemes.
Removed constants: LGBMRegressor; n_estimators = 300/200; learning_rate = 0.05.

Why: the boosting settings and the ensemble weights are hand-set constants. A model with a free parameter cannot be ranked against
persistence in a way that separates skill from a lucky constant, so it may not
produce leaderboard rows (see experiment/RETIRED.md).

The implementation is not deleted from history: it is at commit 2ca48e9,
path ``experiment/E006/``. The artefacts in ``data/`` and ``results/``
stay here, and the leaderboard rows are kept under ``status = 'retired-heuristic'``.
"""
from _exo.retired import RetiredExperiment

EXPERIMENT = "E006"
DIRECTIONS: tuple[str, ...] = ()
REMOVED_MODEL = "LightGBM ensembles with four weighting schemes"
REASON = "the boosting settings and the ensemble weights are hand-set constants"
COMMIT = "2ca48e9"


def main() -> None:
    raise RetiredExperiment(EXPERIMENT, REASON, COMMIT)


if __name__ == "__main__":
    main()
