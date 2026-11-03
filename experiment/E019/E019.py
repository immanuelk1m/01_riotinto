"""E019 — retired 2026-11-03.

Removed model: adaptive-conformal (ACI) deployment rule.
Removed constants: ACI_GAMMA = 0.10; LAMBDA_MAX = 0.5; MIN_LABELS = 13; RECENT = 26.

Why: the step size of the adaptive level is a hand-set constant; it is this rule that held the program's best legacy number (6.160). A model with a free parameter cannot be ranked against
persistence in a way that separates skill from a lucky constant, so it may not
produce leaderboard rows (see experiment/RETIRED.md).

The implementation is not deleted from history: it is at commit 2ca48e9,
path ``experiment/E019/``. The artefacts in ``data/`` and ``results/``
stay here, and the leaderboard rows are kept under ``status = 'retired-heuristic'``.
"""
from _exo.retired import RetiredExperiment

EXPERIMENT = "E019"
DIRECTIONS: tuple[str, ...] = ()
REMOVED_MODEL = "adaptive-conformal (ACI) deployment rule"
REASON = "the step size of the adaptive level is a hand-set constant; it is this rule that held the program's best legacy number (6.160)"
COMMIT = "2ca48e9"


def main() -> None:
    raise RetiredExperiment(EXPERIMENT, REASON, COMMIT)


if __name__ == "__main__":
    main()
