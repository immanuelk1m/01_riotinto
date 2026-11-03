"""A retired experiment must refuse to run; the purge is otherwise cosmetic."""
import importlib

import pytest

from _exo.retired import RetiredExperiment

RETIRED = (
    "E001",
    "E002",
    "E003",
    "E004",
    "E005",
    "E006",
    "E007",
    "E008",
    "E009",
    "E010",
    "E011",
    "E012",
    "E013",
    "E014",
    "E015",
    "E016",
    "E017",
    "E018",
    "E019",
    "E020",
    "E021",
)


@pytest.mark.parametrize("experiment", RETIRED)
def test_retired_experiment_refuses_to_run(experiment):
    module = importlib.import_module(f"experiment.{experiment}.{experiment}")
    assert module.DIRECTIONS == ()
    with pytest.raises(RetiredExperiment):
        module.main()
