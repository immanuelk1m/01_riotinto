"""E014 — retired 2026-11-03.

Removed model: sheet-pool wrapper around the E013 GRU core.
Removed constants: E013 constants; pooling weights.

Why: it wraps the E013 GRU core and adds pooling weights. A model with a free parameter cannot be ranked against
persistence in a way that separates skill from a lucky constant, so it may not
produce leaderboard rows (see experiment/RETIRED.md).

The implementation is not deleted from history: it is at commit 2ca48e9,
path ``experiment/E014/``. The artefacts in ``data/`` and ``results/``
stay here, and the leaderboard rows are kept under ``status = 'retired-heuristic'``.
"""
from _exo.retired import RetiredExperiment

EXPERIMENT = "E014"
DIRECTIONS: tuple[str, ...] = ()
REMOVED_MODEL = "sheet-pool wrapper around the E013 GRU core"
REASON = "it wraps the E013 GRU core and adds pooling weights"
COMMIT = "2ca48e9"


def main() -> None:
    raise RetiredExperiment(EXPERIMENT, REASON, COMMIT)


if __name__ == "__main__":
    main()
