"""E005 — retired 2026-11-03.

Removed model: Ridge over a GridSearchCV penalty grid.
Removed constants: Ridge; GridSearchCV; alpha grid.

Why: the penalty is selected by a grid search, so the reported number is the best point of a searched family. A model with a free parameter cannot be ranked against
persistence in a way that separates skill from a lucky constant, so it may not
produce leaderboard rows (see experiment/RETIRED.md).

The implementation is not deleted from history: it is at commit 2ca48e9,
path ``experiment/E005/``. The artefacts in ``data/`` and ``results/``
stay here, and the leaderboard rows are kept under ``status = 'retired-heuristic'``.
"""
from _exo.retired import RetiredExperiment

EXPERIMENT = "E005"
DIRECTIONS: tuple[str, ...] = ()
REMOVED_MODEL = "Ridge over a GridSearchCV penalty grid"
REASON = "the penalty is selected by a grid search, so the reported number is the best point of a searched family"
COMMIT = "2ca48e9"


def main() -> None:
    raise RetiredExperiment(EXPERIMENT, REASON, COMMIT)


if __name__ == "__main__":
    main()
