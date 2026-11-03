"""E011 — retired 2026-11-03.

Removed model: delayed EWLS correction of E010 GRU forecasts.
Removed constants: EWLS decay and memory-horizon constants on top of E010.

Why: it post-processes GRU forecasts, so it inherits E010's free parameters and adds its own decay constant. A model with a free parameter cannot be ranked against
persistence in a way that separates skill from a lucky constant, so it may not
produce leaderboard rows (see experiment/RETIRED.md).

The implementation is not deleted from history: it is at commit 2ca48e9,
path ``experiment/E011/``. The artefacts in ``data/`` and ``results/``
stay here, and the leaderboard rows are kept under ``status = 'retired-heuristic'``.
"""
from _exo.retired import RetiredExperiment

EXPERIMENT = "E011"
DIRECTIONS: tuple[str, ...] = ()
REMOVED_MODEL = "delayed EWLS correction of E010 GRU forecasts"
REASON = "it post-processes GRU forecasts, so it inherits E010's free parameters and adds its own decay constant"
COMMIT = "2ca48e9"


def main() -> None:
    raise RetiredExperiment(EXPERIMENT, REASON, COMMIT)


if __name__ == "__main__":
    main()
