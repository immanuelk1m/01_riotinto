"""검증된 GP 기준선 기록을 보존하고 타겟 변경 하나만 이어서 학습한다."""
from copy import deepcopy
import argparse
import logging
import os
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import yaml
from src.studies import load_study, spec_diff
from src.study_runner import (digest, file_sha, source_fingerprint, environment, write_json,
                             run_experiment, validate_expected_cells, log_study_run)
from src.study_reporting import compare_forecasts, annual_comparison


def validate_reference(root, parent, manifest, study):
    if manifest['status'] != 'complete':
        raise ValueError('부모 실행이 완료되지 않았다')
    frozen = parent / 'source'
    current = set(p.relative_to(root) for p in (root/'src').rglob('*.py'))
    previous = set(p.relative_to(frozen) for p in (frozen/'src').rglob('*.py'))
    if current != previous:
        raise ValueError('과학 코드 파일 목록이 변경됐다')
    for path in current:
        if file_sha(root/path) != file_sha(frozen/path):
            raise ValueError(f'과학 코드가 변경됐다: {path}')
    if environment() != manifest['context']['environment'] or file_sha(root/'uv.lock') != manifest['context']['uv_lock_sha']:
        raise ValueError('실행 환경이 변경됐다')
    parameters = yaml.safe_load((root/'conf/base/parameters.yml').read_text())
    exog = yaml.safe_load((root/'conf/base/parameters_exogenous.yml').read_text())['exogenous']
    if digest(parameters) != manifest['context']['protocol_sha'] or digest(exog) != manifest['context']['exogenous_config_sha']:
        raise ValueError('평가/외생 설정이 변경됐다')
    for name, sha in manifest['context']['raw_sha'].items():
        if file_sha(root/f'data/01_raw/{name}.parquet') != sha or file_sha(parent/f'inputs/{name}.parquet') != sha:
            raise ValueError('스냅샷이 변경됐다')
    for name, sha in manifest['shared_outputs'].items():
        if file_sha(parent/'shared'/name) != sha:
            raise ValueError('공유 입력이 손상됐다')
    for name, record in manifest['runs'].items():
        if study['experiments'][name] != record['identity']['spec']:
            raise ValueError(f'부모 명세가 변경됐다: {name}')
        if record['status'] != 'complete' or any(file_sha(parent/'runs'/record['run_id']/f) != sha for f,sha in record['outputs'].items()):
            raise ValueError(f'부모 산출물이 손상됐다: {name}')
    if spec_diff(study['experiments']['base_gaussian_process'], study['experiments']['gp_logdiff']) != ['target']:
        raise ValueError('타겟 외 설정 변경')
    return parameters, exog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=ROOT/'data/07_reporting/studies/gp_baseline_v1/d212e5590033bd36')
    args=parser.parse_args();parent=args.parent.resolve()
    logging.getLogger('kedro').setLevel(logging.WARNING);os.environ.setdefault('WANDB_SILENT','true')
    study=load_study(ROOT/'conf/studies/gp_logdiff_v1.yml');old=json.loads((parent/'manifest.json').read_text())
    params,exog=validate_reference(ROOT,parent,old,study)
    context={'source_sha':source_fingerprint(ROOT),'study_sha':digest(study),'parent_execution':str(parent),
             'parent_manifest_sha':file_sha(parent/'manifest.json'),'raw_sha':old['context']['raw_sha'],
             'environment':environment(),'protocol_sha':old['context']['protocol_sha']}
    directory=ROOT/'data/07_reporting/studies/gp_baseline_v1'/digest(context)[:16];directory.mkdir(parents=True,exist_ok=True)
    path=directory/'manifest.json'
    m=json.loads(path.read_text()) if path.exists() else {'study_id':study['study_id'],'context':context,'status':'running','runs':deepcopy(old['runs'])}
    if m['context'] != context:raise ValueError('실행 지문 불일치')
    for folder in ['shared','inputs','source']:
        dest=directory/folder
        if not dest.exists():dest.symlink_to(parent/folder,target_is_directory=True)
    (directory/'continuation_source').mkdir(exist_ok=True)
    extension_source=directory/'continuation_source/run_gp_logdiff.py'
    if not extension_source.exists():extension_source.write_bytes(Path(__file__).read_bytes())
    if file_sha(extension_source)!=file_sha(Path(__file__)):raise ValueError('저장한 continuation 코드 불일치')
    (directory/'runs').mkdir(exist_ok=True)
    for name,record in old['runs'].items():
        dest=directory/'runs'/record['run_id']
        if not dest.exists():dest.symlink_to(parent/'runs'/record['run_id'],target_is_directory=True)
    m['reference_provenance']={name:{'execution':str(parent),'run_id':r['run_id'],'retrained':False} for name,r in old['runs'].items()}
    name='gp_logdiff';spec=study['experiments'][name];base=old['runs']['base_gaussian_process']
    identity={**base['identity'],'experiment':name,'spec':spec,'execution':context,'parent_run_id':base['run_id']}
    rid=digest(identity)[:20];output=directory/'runs'/rid
    r=m['runs'].setdefault(name,{'identity':identity,'run_id':rid,'status':'running','source_experiments':[], 'data_last_week':base['data_last_week']})
    write_json(directory/'study.json',study);write_json(path,m)
    if r['status']!='complete':
        output.mkdir(exist_ok=True)
        print(f'TRAIN gp_logdiff parent={base["run_id"]}',flush=True)
        run_experiment(name,spec,output,parent/'shared',params,exog,{})
        validate_expected_cells(pd.read_parquet(output/'forecasts.parquet'),pd.read_parquet(parent/'shared/targets.parquet'),params['horizon'])
        r.update(status='complete',outputs={f:file_sha(output/f) for f in ['fitted.pkl','training.json','forecasts.parquet','metrics.parquet']})
        write_json(path,m)
    if any(file_sha(output/f)!=sha for f,sha in r['outputs'].items()):raise ValueError('신규 산출물 손상')
    if source_fingerprint(ROOT)!=context['source_sha']:raise ValueError('실행 중 코드 변경')
    fc=pd.read_parquet(output/'forecasts.parquet');cs=[];years=[]
    for role,ref in [('naive','naive'),('root','base_xgb'),('parent','base_gaussian_process'),('additional:drift','drift')]:
        ref_fc=pd.read_parquet(parent/'runs'/old['runs'][ref]['run_id']/'forecasts.parquet')
        c=compare_forecasts(fc,ref_fc,params['ape_cap']).assign(experiment=name,reference=ref,reference_role=role,axis='target')
        c['primary']=c.scope.isin(['overall','series']);c['holm_pvalue']=c.bootstrap_pvalue;c['family_size']=1;c['family_evaluated']=1;cs.append(c)
        years.append(annual_comparison(fc,ref_fc,params['ape_cap']).assign(experiment=name,reference=ref,reference_role=role))
    comparison=pd.concat(cs,ignore_index=True);annual=pd.concat(years,ignore_index=True)
    pd.concat([pd.read_parquet(parent/'comparisons.parquet'),comparison],ignore_index=True).to_parquet(directory/'comparisons.parquet',index=False)
    pd.concat([pd.read_parquet(parent/'annual.parquet'),annual],ignore_index=True).to_parquet(directory/'annual.parquet',index=False)
    metrics=pd.read_parquet(output/'metrics.parquet')
    pd.concat([pd.read_parquet(parent/'summary.parquet'),metrics.assign(experiment=name,run_id=rid,parent='base_gaussian_process',root='base_xgb',axis='target',kind='fit')],ignore_index=True).to_parquet(directory/'summary.parquet',index=False)
    lineage=json.loads((parent/'lineage.json').read_text());lineage.append({'experiment':name,**spec['lineage'],'run_id':rid,'root_run_id':old['runs']['base_xgb']['run_id'],'parent_run_id':base['run_id'],'actual_changes':['target']});write_json(directory/'lineage.json',lineage)
    if not r.get('wandb',{}).get('logged'):
        r['wandb']=log_study_run(name,spec,r,m['runs'],metrics,fc,comparison,annual,'online',directory)
    m.update(status='complete',online_logged=sum(bool(r.get('wandb',{}).get('logged')) for r in m['runs'].values()))
    write_json(path,m);print(f'STUDY_COMPLETE {directory}',flush=True)


if __name__=='__main__':main()
