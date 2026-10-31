import json
from pathlib import Path
base = Path('/content/E019')
direction = json.loads((base/'run_config.json').read_text())['direction']
commodities = ('gold','copper','corn','coffee','wti_crude_oil','wheat')
logs = base/'logs'
names = {p.name for p in (base/'results'/direction).glob('predictions_*.parquet')}
expected = {f'predictions_{c}.parquet' for c in commodities}
complete = names == expected and all((logs/f'{c}_{direction}_{s}').exists() for c in commodities for s in ('transfer.csv','summary.json','protocol.json')) and (logs/f'cross_section_{direction}.csv').exists()
running = False
try:
    pid = int((logs/f'{direction}.pid').read_text())
    running = 'E019.py' in Path(f'/proc/{pid}/cmdline').read_text()
except (FileNotFoundError,ValueError,ProcessLookupError):
    pass
done = (logs/f'DONE_{direction}').exists()
finished = done and complete and not (logs/f'FAILED_{direction}').exists()
failed = (logs/f'FAILED_{direction}').exists() or (done and not complete) or (not running and not finished)
print(f'STATUS {direction} done={len(names & expected)}/6 running={running} finished={finished} failed={failed}')
out=logs/f'run_{direction}.out'
if out.exists(): print('\n'.join(out.read_text().splitlines()[-8:]))
