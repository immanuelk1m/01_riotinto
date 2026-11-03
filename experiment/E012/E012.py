"""E012 — retired 2026-11-03.

Removed model: delayed fixed-share aggregation of E009/E010 forecasts.
Removed constants: fixed-share learning rate and delay constants.

Why: the aggregation rate is a hand-set constant applied to knob-carrying inputs. A model with a free parameter cannot be ranked against
persistence in a way that separates skill from a lucky constant, so it may not
produce leaderboard rows (see experiment/RETIRED.md).

The implementation is not deleted from history: it is at commit 2ca48e9,
path ``experiment/E012/``. The artefacts in ``data/`` and ``results/``
stay here, and the leaderboard rows are kept under ``status = 'retired-heuristic'``.
"""
from _exo.retired import RetiredExperiment

EXPERIMENT = "E012"
DIRECTIONS: tuple[str, ...] = ()
REMOVED_MODEL = "delayed fixed-share aggregation of E009/E010 forecasts"
REASON = "the aggregation rate is a hand-set constant applied to knob-carrying inputs"
COMMIT = "2ca48e9"


def main() -> None:
    raise RetiredExperiment(EXPERIMENT, REASON, COMMIT)


if __name__ == "__main__":
    main()
