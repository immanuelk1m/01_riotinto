"""E016 — retired 2026-11-03.

Removed model: affine calibration of E014 forecasts.
Removed constants: recent-26 and static-52 calibration windows on top of E014.

Why: it calibrates E014's GRU output and chooses the calibration window by hand. A model with a free parameter cannot be ranked against
persistence in a way that separates skill from a lucky constant, so it may not
produce leaderboard rows (see experiment/RETIRED.md).

The implementation is not deleted from history: it is at commit 2ca48e9,
path ``experiment/E016/``. The artefacts in ``data/`` and ``results/``
stay here, and the leaderboard rows are kept under ``status = 'retired-heuristic'``.
"""
from _exo.retired import RetiredExperiment

EXPERIMENT = "E016"
DIRECTIONS: tuple[str, ...] = ()
REMOVED_MODEL = "affine calibration of E014 forecasts"
REASON = "it calibrates E014's GRU output and chooses the calibration window by hand"
COMMIT = "2ca48e9"


def main() -> None:
    raise RetiredExperiment(EXPERIMENT, REASON, COMMIT)


if __name__ == "__main__":
    main()
