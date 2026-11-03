"""E021 — retired 2026-11-03.

Removed model: multi-output ridge over 730 series.
Removed constants: RIDGE_LAMBDA = 1.0; LAGS = 4; MIN_WEEKS = 250.

Why: the ridge penalty is a constant, and this single experiment owns 53 % of the leaderboard rows. A model with a free parameter cannot be ranked against
persistence in a way that separates skill from a lucky constant, so it may not
produce leaderboard rows (see experiment/RETIRED.md).

The implementation is not deleted from history: it is at commit 2ca48e9,
path ``experiment/E021/``. The artefacts in ``data/`` and ``results/``
stay here, and the leaderboard rows are kept under ``status = 'retired-heuristic'``.
"""
from _exo.retired import RetiredExperiment

EXPERIMENT = "E021"
DIRECTIONS: tuple[str, ...] = ()
REMOVED_MODEL = "multi-output ridge over 730 series"
REASON = "the ridge penalty is a constant, and this single experiment owns 53 % of the leaderboard rows"
COMMIT = "2ca48e9"


def main() -> None:
    raise RetiredExperiment(EXPERIMENT, REASON, COMMIT)


if __name__ == "__main__":
    main()
