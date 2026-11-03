"""E020 — retired 2026-11-03.

Removed model: GRU with the E014 snapshot.
Removed constants: GRU; snapshot constants.

Why: it runs the GRU core against a fixed snapshot. A model with a free parameter cannot be ranked against
persistence in a way that separates skill from a lucky constant, so it may not
produce leaderboard rows (see experiment/RETIRED.md).

The implementation is not deleted from history: it is at commit 2ca48e9,
path ``experiment/E020/``. The artefacts in ``data/`` and ``results/``
stay here, and the leaderboard rows are kept under ``status = 'retired-heuristic'``.
"""
from _exo.retired import RetiredExperiment

EXPERIMENT = "E020"
DIRECTIONS: tuple[str, ...] = ()
REMOVED_MODEL = "GRU with the E014 snapshot"
REASON = "it runs the GRU core against a fixed snapshot"
COMMIT = "2ca48e9"


def main() -> None:
    raise RetiredExperiment(EXPERIMENT, REASON, COMMIT)


if __name__ == "__main__":
    main()
