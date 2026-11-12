from copy import deepcopy
import pytest
from src.studies import load_study, spec_diff, validate_study


def test_logdiff_changes_only_target_and_keeps_reference_specs():
    old=load_study('conf/studies/gp_baseline_v1.yml')
    new=load_study('conf/studies/gp_logdiff_v1.yml')
    assert len(new['experiments'])==7
    for name,spec in old['experiments'].items():
        assert new['experiments'][name]==spec
    child=new['experiments']['gp_logdiff']
    assert child['lineage']['parent']=='base_gaussian_process'
    assert child['target']=='logdiff'
    assert spec_diff(old['experiments']['base_gaussian_process'],child)==['target']


def test_unrelated_kernel_change_is_rejected():
    new=deepcopy(load_study('conf/studies/gp_logdiff_v1.yml'))
    new['experiments']['gp_logdiff']['model_params']['nu']=2.5
    with pytest.raises(ValueError,match='changed_paths'):
        validate_study(new)
