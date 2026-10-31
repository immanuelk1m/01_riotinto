"""Bounded Colab CLI subprocesses for E020; GPU session, artifacts validated locally."""
import argparse
import re
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
NAME = 'E020'


def call(*args, timeout=60, attempts=5):
    for attempt in range(attempts):
        try:
            result = subprocess.run(args, cwd=BASE, text=True, capture_output=True, timeout=timeout, check=True)
            return result.stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt + 1 == attempts:
                raise RuntimeError(f'Command failed: {args[0:3]}') from None
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', required=True)
    parser.add_argument('--direction', required=True, choices=('base', 'recent', 'tta'))
    args = parser.parse_args()
    s, d = args.session, args.direction
    remote = f'/content/{NAME}'
    status = call('colab', 'status')
    if f'[{s}] ' not in status:
        call('colab', 'new', '-s', s, '--gpu', 'T4', timeout=180, attempts=20)
    try:
        call('colab', 'exec', '-s', s, '-f', 'remote/bootstrap.py')
        # transparent execution: -f reads these locally and sends them to the kernel
        call('colab', 'upload', '-s', s, f'{NAME}.py', f'{remote}/{NAME}.py')
        call('colab', 'upload', '-s', s, f'test_{NAME.lower()}.py', f'{remote}/test_{NAME.lower()}.py')
        call('colab', 'upload', '-s', s, f'configs/{d}.json', f'{remote}/run_config.json')
        for path in sorted((BASE.parent / 'E014' / 'data').glob('*.parquet')):
            call('colab', 'upload', '-s', s, str(path), f'{remote}/data/{path.name}')
        for path in sorted((BASE.parent / 'E014' / 'features').glob('*.json')):
            call('colab', 'upload', '-s', s, str(path), f'{remote}/features/{path.name}')
        print(call('colab', 'exec', '-s', s, '-f', 'remote/smoke.py'), flush=True)
        print(call('colab', 'exec', '-s', s, '-f', 'remote/launch.py', attempts=1), flush=True)
        for _ in range(480):
            out = call('colab', 'exec', '-s', s, '-f', 'remote/poll.py', timeout=120)
            print(out, flush=True)
            if 'finished=True' in out or 'failed=True' in out:
                break
            time.sleep(30)
        else:
            raise RuntimeError('Polling exceeded 480 rounds')
        listing = call('colab', 'exec', '-s', s, '-f', 'remote/list.py', timeout=120)
        files = [ln.strip() for ln in re.sub(r'\x1b\[[0-9;]*m', '', listing).splitlines() if ln.startswith(remote + '/')]
        if not files:
            raise RuntimeError('No downloadable artifacts')
        for path in files:
            relative = Path(path).relative_to(remote)
            if '..' in relative.parts:
                raise ValueError('Unsafe artifact path')
            local = BASE / relative
            local.parent.mkdir(parents=True, exist_ok=True)
            call('colab', 'download', '-s', s, path, str(local), timeout=180)
        if 'failed=True' in out:
            raise RuntimeError('Remote run failed; logs downloaded')
        print(call('uv', 'run', 'python', f'{NAME}.py', 'leaderboard', d, timeout=300), flush=True)
    except Exception:
        print(f'Preserving Colab session {s} for recovery', flush=True)
        raise
    call('colab', 'stop', '-s', s)
    print(f'{NAME} {d} verified and VM stopped', flush=True)


if __name__ == '__main__':
    main()
