import json
from pathlib import Path
base=Path('/content/E019')
direction=json.loads((base/'run_config.json').read_text())['direction']
for folder in (base/'results'/direction,base/'logs'):
    for path in sorted(folder.glob('*')):
        if path.is_file() and path.suffix != '.pid': print(path)
