from pathlib import Path
for name in ('inputs', 'logs', 'results'):
    (Path('/content/E016') / name).mkdir(parents=True, exist_ok=True)
print('ready /content/E016')
