"""E002 — retired 2026-11-03.

Removed model: LightGBM with a top-k feature screen.
Removed constants: LGBMRegressor; TOP_K = 30.

Why: TOP_K decides which features enter the model and was never chosen by a rule. A model with a free parameter cannot be ranked against
persistence in a way that separates skill from a lucky constant, so it may not
produce leaderboard rows (see experiment/RETIRED.md).

The implementation is not deleted from history: it is at commit 2ca48e9,
path ``experiment/E002/``. The artefacts in ``data/`` and ``results/``
stay here, and the leaderboard rows are kept under ``status = 'retired-heuristic'``.
"""
from _exo.retired import RetiredExperiment

EXPERIMENT = "E002"
DIRECTIONS: tuple[str, ...] = ()
REMOVED_MODEL = "LightGBM with a top-k feature screen"
REASON = "TOP_K decides which features enter the model and was never chosen by a rule"
COMMIT = "2ca48e9"


def main() -> None:
    raise RetiredExperiment(EXPERIMENT, REASON, COMMIT)


if __name__ == "__main__":
    main()
