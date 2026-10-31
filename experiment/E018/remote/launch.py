import json
import os
import subprocess
from pathlib import Path
base = Path('/content/E018')
direction = json.loads((base/'run_config.json').read_text())['direction']
if direction not in ('eproc', 'cs_weight', 'smcs'):
    raise ValueError(direction)
logs = base/'logs'
pidfile = logs/f'{direction}.pid'
if pidfile.exists() or list((base/'results'/direction).glob('*')):
    raise RuntimeError('Existing artifacts; inspect before relaunch')
command = f'python -u E018.py {direction} > logs/run_{direction}.out 2>&1; rc=$?; if [ $rc -eq 0 ]; then touch logs/DONE_{direction}; else echo $rc > logs/FAILED_{direction}; fi; exit $rc'
p = subprocess.Popen(['bash','-c',command],cwd=base,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'2'})
pidfile.write_text(str(p.pid))
print(f'launched E018 {direction} pid={p.pid}')
