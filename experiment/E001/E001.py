"""E001 — retired 2026-11-03.

Removed model: LightGBM + genetic feature search.
Removed constants: LGBMRegressor; array_split; rng.choice; GENERATIONS; PRUNE_FRAC; N_KEEP.

Why: the forecast comes from boosted trees whose depth, learning rate and tree count were hand-set, on top of a stochastic feature search. A model with a free parameter cannot be ranked against
persistence in a way that separates skill from a lucky constant, so it may not
produce leaderboard rows (see experiment/RETIRED.md).

The implementation is not deleted from history: it is at commit 2ca48e9,
path ``experiment/E001/``. The artefacts in ``data/`` and ``results/``
stay here, and the leaderboard rows are kept under ``status = 'retired-heuristic'``.
"""
from _exo.retired import RetiredExperiment

EXPERIMENT = "E001"
DIRECTIONS: tuple[str, ...] = ()
REMOVED_MODEL = "LightGBM + genetic feature search"
REASON = "the forecast comes from boosted trees whose depth, learning rate and tree count were hand-set, on top of a stochastic feature search"
COMMIT = "2ca48e9"


def main() -> None:
    raise RetiredExperiment(EXPERIMENT, REASON, COMMIT)


if __name__ == "__main__":
    main()
