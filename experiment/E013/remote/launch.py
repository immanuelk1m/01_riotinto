"""Upload E013.py, data/, features/ and optional run_config.json first."""
import json
import os
import subprocess
from pathlib import Path

base = Path('/content/E013')
config = base / 'run_config.json'
direction = json.loads(config.read_text()).get('direction', 'inv01') if config.exists() else 'inv01'
if direction not in ('erm', 'inv01', 'inv1'):
    raise ValueError('Invalid direction')
logs = base / 'logs'
logs.mkdir(exist_ok=True)
pidfile = logs / f'{direction}.pid'
if pidfile.exists() or list((base / 'results' / direction).glob('*')):
    raise RuntimeError('Existing run artifacts: inspect and use a fresh session')
command = f'python -u E013.py {direction} > logs/run_{direction}.out 2>&1; rc=$?; if [ $rc -eq 0 ]; then touch logs/DONE_{direction}; else echo $rc > logs/FAILED_{direction}; fi; exit $rc'
process = subprocess.Popen(['bash', '-c', command], cwd=base, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, start_new_session=True,
                           env={**os.environ, 'OMP_NUM_THREADS': '2', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
pidfile.write_text(str(process.pid))
print(f'launched E013 {direction} pid={process.pid}')
