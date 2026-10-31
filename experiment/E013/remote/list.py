import json
from pathlib import Path

base = Path('/content/E013')
config = base / 'run_config.json'
direction = json.loads(config.read_text()).get('direction', 'inv01') if config.exists() else 'inv01'
if direction not in ('erm', 'inv01', 'inv1'):
    raise ValueError('Invalid direction')
for folder in (base / 'results' / direction, base / 'logs'):
    for path in sorted(folder.glob('*')):
        if path.is_file() and path.suffix != '.pid':
            print(path)
