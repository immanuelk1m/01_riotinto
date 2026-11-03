"""Exceptions raised by a retired experiment.

A retired experiment is not deleted: its artefacts stay in ``experiment/E0xx/``
and the implementation stays in git at the commit named in the stub.  What is
removed is the ability to *run* it, because its model carries free parameters and
therefore cannot contribute a row to the leaderboard.
"""

from __future__ import annotations


class RetiredExperiment(RuntimeError):
    """Raised when someone tries to run an experiment whose model was purged."""

    def __init__(self, experiment: str, reason: str, commit: str = ""):
        self.experiment = experiment
        self.reason = reason
        self.commit = commit
        location = f" The implementation is at commit {commit}." if commit else ""
        super().__init__(
            f"{experiment} is retired: {reason}. It cannot produce leaderboard rows, "
            f"because a model with a free parameter cannot be ranked against persistence "
            f"in a way that separates skill from a lucky constant.{location}"
        )


__all__ = ["RetiredExperiment"]
