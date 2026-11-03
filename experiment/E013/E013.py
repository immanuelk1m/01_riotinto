"""E013 — retired 2026-11-03.

Removed model: invariant-risk GRU core.
Removed constants: GRU; environment split; penalty weight.

Why: the invariance penalty and the network are both free parameters. A model with a free parameter cannot be ranked against
persistence in a way that separates skill from a lucky constant, so it may not
produce leaderboard rows (see experiment/RETIRED.md).

The implementation is not deleted from history: it is at commit 2ca48e9,
path ``experiment/E013/``. The artefacts in ``data/`` and ``results/``
stay here, and the leaderboard rows are kept under ``status = 'retired-heuristic'``.
"""
from _exo.retired import RetiredExperiment

EXPERIMENT = "E013"
DIRECTIONS: tuple[str, ...] = ()
REMOVED_MODEL = "invariant-risk GRU core"
REASON = "the invariance penalty and the network are both free parameters"
COMMIT = "2ca48e9"


def main() -> None:
    raise RetiredExperiment(EXPERIMENT, REASON, COMMIT)


if __name__ == "__main__":
    main()
