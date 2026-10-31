import json
from pathlib import Path

base = Path('/content/E013')
config = base / 'run_config.json'
direction = json.loads(config.read_text()).get('direction', 'inv01') if config.exists() else 'inv01'
if direction not in ('erm', 'inv01', 'inv1'):
    raise ValueError('Invalid direction')
logs = base / 'logs'
commodities = ('gold', 'copper', 'corn', 'coffee', 'wti_crude_oil')
names = {p.name for p in (base / 'results' / direction).glob('predictions_*.parquet')}
expected = {f'predictions_{c}.parquet' for c in commodities}
complete = names == expected and all((logs / f'{c}_{direction}_{suffix}').exists()
    for c in commodities for suffix in ('curves.json', 'transfer.csv', 'summary.json'))
running = False
try:
    pid = int((logs / f'{direction}.pid').read_text())
    running = 'E013.py' in Path(f'/proc/{pid}/cmdline').read_text()
except (ValueError, FileNotFoundError, ProcessLookupError):
    pass
done = (logs / f'DONE_{direction}').exists()
failed_marker = (logs / f'FAILED_{direction}').exists()
finished = done and complete and not failed_marker
failed = failed_marker or (done and not complete) or (not running and not finished)
print(f'STATUS {direction} done={len(names & expected)}/5 running={running} finished={finished} failed={failed}')
out = logs / f'run_{direction}.out'
if out.exists():
    print('\n'.join(out.read_text().splitlines()[-8:]))
