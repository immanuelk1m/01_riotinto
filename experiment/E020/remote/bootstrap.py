from pathlib import Path
for name in ('data', 'features', 'logs', 'results'):
    (Path('/content/E020') / name).mkdir(parents=True, exist_ok=True)
print('ready /content/E020')
