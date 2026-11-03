"""E018 — retired 2026-11-03.

Removed model: e-process with shrunk cross-sectional weights.
Removed constants: ALPHA = 0.05; MIN_LABELS = 13; SHRINK_MIN = 13; SHRINK_FRACTION = 0.5; EFFECT_SCALE = 0.01.

Why: five hand-set constants define the rule, including a shrinkage schedule. A model with a free parameter cannot be ranked against
persistence in a way that separates skill from a lucky constant, so it may not
produce leaderboard rows (see experiment/RETIRED.md).

The implementation is not deleted from history: it is at commit 2ca48e9,
path ``experiment/E018/``. The artefacts in ``data/`` and ``results/``
stay here, and the leaderboard rows are kept under ``status = 'retired-heuristic'``.
"""
from _exo.retired import RetiredExperiment

EXPERIMENT = "E018"
DIRECTIONS: tuple[str, ...] = ()
REMOVED_MODEL = "e-process with shrunk cross-sectional weights"
REASON = "five hand-set constants define the rule, including a shrinkage schedule"
COMMIT = "2ca48e9"


def main() -> None:
    raise RetiredExperiment(EXPERIMENT, REASON, COMMIT)


if __name__ == "__main__":
    main()
