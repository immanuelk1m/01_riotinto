"""Bounded Colab CLI subprocesses; preserve VM until artifacts validate locally."""
import argparse
import re
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent


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
    parser=argparse.ArgumentParser()
    parser.add_argument('--session', required=True)
    parser.add_argument('--direction', required=True, choices=('static52','recent26','drift26'))
    args=parser.parse_args()
    s,d=args.session,args.direction
    # No GPU needed for the six small affine replays.
    status=call('colab','status')
    if f'[{s}] ' not in status:
        call('colab','new','-s',s,timeout=120,attempts=20)
    try:
        call('colab','exec','-s',s,'-f','remote/bootstrap.py')
        for filename in ('E016.py','test_e016.py'):
            call('colab','upload','-s',s,filename,f'/content/E016/{filename}')
        call('colab','upload','-s',s,f'configs/{d}.json','/content/E016/run_config.json')
        for path in sorted((BASE.parent/'E014'/'results'/'erm').glob('predictions_*.parquet')):
            call('colab','upload','-s',s,str(path),f'/content/E016/inputs/{path.name}')
        print(call('colab','exec','-s',s,'-f','remote/smoke.py'),flush=True)
        # A launch retry is unsafe; inspect remote PID/artifacts if acknowledgement is lost.
        print(call('colab','exec','-s',s,'-f','remote/launch.py',attempts=1),flush=True)
        for _ in range(240):
            out=call('colab','exec','-s',s,'-f','remote/poll.py')
            print(out,flush=True)
            if 'finished=True' in out or 'failed=True' in out: break
            time.sleep(10)
        else:
            raise RuntimeError('Polling exceeded 240 rounds')
        listing=call('colab','exec','-s',s,'-f','remote/list.py')
        files=[line.strip() for line in re.sub(r'\x1b\[[0-9;]*m','',listing).splitlines() if line.startswith('/content/E016/')]
        if not files: raise RuntimeError('No downloadable artifacts')
        for remote in files:
            relative=Path(remote).relative_to('/content/E016')
            if '..' in relative.parts: raise ValueError('Unsafe artifact path')
            local=BASE/relative
            local.parent.mkdir(parents=True,exist_ok=True)
            call('colab','download','-s',s,remote,str(local))
        if 'failed=True' in out: raise RuntimeError('Remote run failed; logs downloaded')
        print(call('uv','run','python','E016.py','leaderboard',d,timeout=120),flush=True)
    except Exception:
        print(f'Preserving Colab session {s} for recovery',flush=True)
        raise
    call('colab','stop','-s',s)
    print(f'E016 {d} verified and VM stopped',flush=True)


if __name__=='__main__': main()
