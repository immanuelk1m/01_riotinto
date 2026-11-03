"""E008 — retired 2026-11-03.

Removed model: LightGBM loss/averaging variants.
Removed constants: LGBMRegressor; huber/dro/global variants.

Why: the ranking across variants is decided by the loss and averaging constants. A model with a free parameter cannot be ranked against
persistence in a way that separates skill from a lucky constant, so it may not
produce leaderboard rows (see experiment/RETIRED.md).

The implementation is not deleted from history: it is at commit 2ca48e9,
path ``experiment/E008/``. The artefacts in ``data/`` and ``results/``
stay here, and the leaderboard rows are kept under ``status = 'retired-heuristic'``.
"""
from _exo.retired import RetiredExperiment

EXPERIMENT = "E008"
DIRECTIONS: tuple[str, ...] = ()
REMOVED_MODEL = "LightGBM loss/averaging variants"
REASON = "the ranking across variants is decided by the loss and averaging constants"
COMMIT = "2ca48e9"


def main() -> None:
    raise RetiredExperiment(EXPERIMENT, REASON, COMMIT)


if __name__ == "__main__":
    main()
