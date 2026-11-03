"""E010 — retired 2026-11-03.

Removed model: GRU (torch) with registration.
Removed constants: GRU; hidden size; learning rate; epochs; patience.

Why: same GRU core as E009 with a different regularisation setting. A model with a free parameter cannot be ranked against
persistence in a way that separates skill from a lucky constant, so it may not
produce leaderboard rows (see experiment/RETIRED.md).

The implementation is not deleted from history: it is at commit 2ca48e9,
path ``experiment/E010/``. The artefacts in ``data/`` and ``results/``
stay here, and the leaderboard rows are kept under ``status = 'retired-heuristic'``.
"""
from _exo.retired import RetiredExperiment

EXPERIMENT = "E010"
DIRECTIONS: tuple[str, ...] = ()
REMOVED_MODEL = "GRU (torch) with registration"
REASON = "same GRU core as E009 with a different regularisation setting"
COMMIT = "2ca48e9"


def main() -> None:
    raise RetiredExperiment(EXPERIMENT, REASON, COMMIT)


if __name__ == "__main__":
    main()
