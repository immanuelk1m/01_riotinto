import json
import os
import subprocess
from pathlib import Path
base = Path('/content/E016')
direction = json.loads((base/'run_config.json').read_text())['direction']
if direction not in ('static52', 'recent26', 'drift26'):
    raise ValueError(direction)
logs = base/'logs'
pidfile = logs/f'{direction}.pid'
if pidfile.exists() or list((base/'results'/direction).glob('*')):
    raise RuntimeError('Existing artifacts; inspect before relaunch')
command = f'python -u E016.py {direction} > logs/run_{direction}.out 2>&1; rc=$?; if [ $rc -eq 0 ]; then touch logs/DONE_{direction}; else echo $rc > logs/FAILED_{direction}; fi; exit $rc'
p = subprocess.Popen(['bash','-c',command],cwd=base,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'2'})
pidfile.write_text(str(p.pid))
print(f'launched E016 {direction} pid={p.pid}')
