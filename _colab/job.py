"""Run this repository's experiments on a Colab VM instead of the Mac mini CPU.

Policy (see ``_colab/README.md``): research compute runs on Colab. The Mac mini
keeps only orchestration (this module) and the Postgres-backed ``prepare`` step.

Typical use, from the repository root on the Mac mini::

    uv run python -m _colab.job run --exp E021 --cmd "ridge" --cmd "ebh" --cmd "pooled"
    uv run python -m _colab.job run --exp E017 --cmd "gold" --gpu T4
    uv run python -m _colab.job status
    uv run python -m _colab.job stop --session e021-smoke
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'experiment'
ANSI = re.compile(r'\x1b\[[0-9;]*m')
OK = 'COLAB-HARNESS-OK'
SKIP_IN_BUNDLE = {'remote', 'logs', 'results', 'colab_runs', '__pycache__', '.ipynb_checkpoints'}


# --------------------------------------------------------------------------- CLI plumbing

def clean(text: str) -> str:
    return ANSI.sub('', text or '')


def call(*args: str, timeout: float = 180, attempts: int = 3) -> str:
    """Run a colab CLI command, retrying transient failures. Returns cleaned stdout."""
    last = ''
    for attempt in range(attempts):
        try:
            proc = subprocess.run(list(args), text=True, capture_output=True, timeout=timeout)
            if proc.returncode == 0:
                return clean(proc.stdout)
            last = clean(proc.stdout + proc.stderr).strip()
        except subprocess.TimeoutExpired:
            last = f'timeout after {timeout}s'
        if attempt + 1 < attempts:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f'{" ".join(args[:3])} failed: {last}')


def session_exists(session: str) -> bool:
    # NOTE: `colab status` exits 0 even for unknown sessions, so the text decides.
    out = call('colab', 'status', '-s', session)
    return 'not found' not in out and f'[{session}]' in out


def exec_code(session: str, code: str, timeout: float = 180, expect: bool = True,
              attempts: int = 3) -> str:
    """Execute Python on the VM kernel via stdin.

    ``expect=True`` requires the OK sentinel; polling passes ``expect=False`` so a
    single stalled round cannot kill a job that is still making progress remotely.
    """
    ast.parse(code)  # a malformed snippet must fail here, not as a remote SyntaxError
    last = ''
    for attempt in range(attempts):
        try:
            proc = subprocess.run(['colab', 'exec', '-s', session, '--timeout', str(timeout)],
                                  input=code, text=True, capture_output=True, timeout=timeout + 90)
        except subprocess.TimeoutExpired:
            last = f'timeout after {timeout + 90}s'
            time.sleep(10 * (attempt + 1))
            continue
        out = clean(proc.stdout + proc.stderr)
        if not expect or OK in out:
            return out
        last = out[-1500:]
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f'remote execution failed: {last}')


def ensure_session(session: str, gpu: str) -> None:
    if session_exists(session):
        print(f'[harness] reusing session {session}', flush=True)
        return
    args = ['colab', 'new', '-s', session]
    if gpu and gpu.lower() != 'none':
        args += ['--gpu', gpu]
    print(f'[harness] creating session {session} (gpu={gpu or "none"})', flush=True)
    call(*args, timeout=300, attempts=20)


# --------------------------------------------------------------------------- remote snippets

def stage_code(exp: str, remote: str) -> str:
    return f'''
import pathlib, tarfile
base = pathlib.Path({remote!r})
base.mkdir(parents=True, exist_ok=True)
with tarfile.open('/content/_bundle.tar.gz') as tf:
    tf.extractall(base, filter='data')
for name in ('logs', 'results'):
    (base / name).mkdir(exist_ok=True)
print({OK!r}, 'staged', len(list(base.rglob("*"))), 'entries')
'''


def launch_code(remote: str, exp: str, cmd: str, tag: str) -> str:
    return f'''
import os, pathlib, subprocess
remote, exp, cmd, tag = {remote!r}, {exp!r}, {cmd!r}, {tag!r}
base = pathlib.Path(remote)
done, failed = base / 'logs' / f'DONE_{{tag}}', base / 'logs' / f'FAILED_{{tag}}'
if done.exists() or failed.exists():
    raise RuntimeError('stale sentinel for ' + tag)
shell = (f'python -u {{exp}}.py {{cmd}} > logs/run_{{tag}}.out 2>&1; rc=$?; '
         f'if [ $rc -eq 0 ]; then touch logs/DONE_{{tag}}; else echo $rc > logs/FAILED_{{tag}}; fi')
process = subprocess.Popen(['bash', '-c', shell], cwd=base, start_new_session=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           env={{**os.environ, 'OMP_NUM_THREADS': '2'}})
print({OK!r}, 'launched', tag, 'pid', process.pid)
'''


def poll_code(remote: str, tag: str) -> str:
    return f'''
import pathlib
base, tag = pathlib.Path({remote!r}), {tag!r}
done, failed = (base / 'logs' / f'DONE_{{tag}}').exists(), (base / 'logs' / f'FAILED_{{tag}}').exists()
tail = ''
out = base / 'logs' / f'run_{{tag}}.out'
if out.exists():
    tail = '\\n'.join(out.read_text(errors='replace').splitlines()[-6:])
state = 'done' if done else ('failed' if failed else 'running')
print({OK!r}, f'state={{state}} tag={{tag}}')
print(tail)
'''


def list_code(remote: str) -> str:
    keep = ('logs', 'results')
    return f'''
import pathlib
base = pathlib.Path({remote!r})
keep = {keep!r}
files = [p for p in sorted(base.rglob('*')) if p.is_file()
         and any(part in keep for part in p.relative_to(base).parts)
         and not p.name.startswith('DONE_') and not p.name.startswith('FAILED_')
         and p.suffix != '.pid']
print({OK!r})
for path in files:
    print('ARTIFACT', path)
'''


# --------------------------------------------------------------------------- bundle

def build_bundle(exp: str, extra: list[str], data_from: str | None = None) -> Path:
    source = EXPERIMENT / exp
    if not source.is_dir():
        raise SystemExit(f'no such experiment: {source}')
    script = source / f'{exp}.py'
    if not script.exists():
        raise SystemExit(f'missing script: {script}')
    handle = tempfile.NamedTemporaryFile(prefix=f'{exp}-bundle-', suffix='.tar.gz', delete=False)
    handle.close()
    with tarfile.open(handle.name, 'w:gz') as tf:
        for path in sorted(source.rglob('*')):
            relative = path.relative_to(source)
            if set(relative.parts) & SKIP_IN_BUNDLE:
                continue
            if path.is_file():
                tf.add(path, arcname=str(relative))
        if data_from:
            donor = EXPERIMENT / data_from
            if not donor.is_dir():
                raise SystemExit(f'no such --data-from experiment: {donor}')
            for folder in ('data', 'features'):
                for path in sorted((donor / folder).glob('*')):
                    if path.is_file():
                        tf.add(path, arcname=f'{folder}/{path.name}')
        for name in extra:
            path = (ROOT / name) if not Path(name).is_absolute() else Path(name)
            if not path.exists():
                raise SystemExit(f'missing --include path: {name}')
            tf.add(path, arcname=str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.name))
    size = Path(handle.name).stat().st_size / 1e6
    print(f'[harness] bundle {size:.1f} MB', flush=True)
    if size > 800:
        print('[harness] WARNING: bundle is large; consider a Drive mount instead', flush=True)
    return Path(handle.name)


def fetch(session: str, remote: str, base: Path) -> int:
    """Download every artifact under the remote logs/ and results/ trees."""
    listing = exec_code(session, list_code(remote))
    paths = [ln.split(' ', 1)[1].strip() for ln in listing.splitlines() if ln.startswith('ARTIFACT ')]
    for path in paths:
        relative = Path(path).relative_to(remote)
        if '..' in relative.parts:
            raise ValueError(f'unsafe artifact path: {path}')
        target = base / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        call('colab', 'download', '-s', session, path, str(target), timeout=600)
    print(f'[harness] downloaded {len(paths)} artifacts -> {base}', flush=True)
    return len(paths)


# --------------------------------------------------------------------------- run

def run_job(args: argparse.Namespace) -> int:
    tag = args.tag or dt.datetime.now().strftime('%H%M%S')
    session = args.session or f'{args.exp.lower()}-{tag}'
    remote = f'/content/{args.exp}'
    bundle = build_bundle(args.exp, args.include, args.data_from)
    started = time.time()
    ok = False
    ensure_session(session, args.gpu)
    try:
        if args.install:
            print(f'[harness] installing {args.install}', flush=True)
            call('colab', 'install', '-s', session, *args.install, timeout=600)
        call('colab', 'upload', '-s', session, str(bundle), '/content/_bundle.tar.gz', timeout=900)
        print(exec_code(session, stage_code(args.exp, remote)).strip(), flush=True)
        base = EXPERIMENT / args.exp / 'colab_runs' / tag
        for index, cmd in enumerate(args.cmd, start=1):
            tag_cmd = f'{tag}_{index}'
            print(f'[harness] launch [{index}/{len(args.cmd)}] {args.exp}.py {cmd}', flush=True)
            exec_code(session, launch_code(remote, args.exp, cmd, tag_cmd))
            deadline = time.time() + args.timeout
            while True:
                time.sleep(args.poll)
                state = exec_code(session, poll_code(remote, tag_cmd), expect=False)
                if OK not in state:
                    print(f'[harness] poll round unreadable, continuing '
                          f'({(time.time() - started) / 60:.1f} min)', flush=True)
                    if time.time() > deadline:
                        raise RuntimeError(f'timeout after {args.timeout}s: {cmd}')
                    continue
                tail = state.strip().splitlines()[1:]
                print(f'[harness] {state.strip().splitlines()[0].replace(OK, "").strip()} '
                      f'({(time.time() - started) / 60:.1f} min)', flush=True)
                if 'state=done' in state:
                    break
                if 'state=failed' in state:
                    print('\n'.join(tail), flush=True)
                    raise RuntimeError(f'remote run failed: {cmd}')
                if time.time() > deadline:
                    raise RuntimeError(f'timeout after {args.timeout}s: {cmd}')
        fetch(session, remote, base)
        ok = True
    finally:
        Path(bundle).unlink(missing_ok=True)
        # Never abandon finished work: salvage artifacts before releasing the VM.
        if not ok and session_exists(session):
            try:
                base = EXPERIMENT / args.exp / 'colab_runs' / tag
                print(f'[harness] run incomplete; salvaging artifacts -> {base}', flush=True)
                fetch(session, remote, base)
            except Exception as exc:  # noqa: BLE001 - salvage is best effort
                print(f'[harness] WARNING: salvage failed: {exc}', flush=True)
        if args.keep:
            print(f'[harness] session kept: {session} (colab stop -s {session})', flush=True)
        elif session_exists(session):
            try:
                call('colab', 'stop', '-s', session, attempts=2)
                print(f'[harness] session stopped: {session}', flush=True)
            except RuntimeError as exc:
                print(f'[harness] WARNING: stop failed: {exc} (colab stop -s {session})', flush=True)
    if ok and args.promote:
        base = EXPERIMENT / args.exp / 'colab_runs' / tag
        for folder in ('logs', 'results'):
            if (base / folder).is_dir():
                target = EXPERIMENT / args.exp / folder
                target.mkdir(exist_ok=True)
                for path in (base / folder).rglob('*'):
                    if path.is_file():
                        destination = target / path.relative_to(base / folder)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(path.read_bytes())
        print(f'[harness] promoted artifacts into experiment/{args.exp}/{{logs,results}}', flush=True)
    print(f'[harness] {"finished" if ok else "aborted"} in {(time.time() - started) / 60:.1f} min', flush=True)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='_colab.job', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    run = sub.add_parser('run', help='bundle an experiment, run it on a Colab VM, fetch artifacts')
    run.add_argument('--exp', required=True, help='experiment id, e.g. E021')
    run.add_argument('--cmd', action='append', required=True, help='script arguments, repeatable')
    run.add_argument('--gpu', default='none', help='none|T4|L4|A100|H100 (default: none)')
    run.add_argument('--install', nargs='*', default=[], help='packages to install on the VM')
    run.add_argument('--include', action='append', default=[], help='extra repo paths to bundle')
    run.add_argument('--data-from', default=None,
                     help='donor experiment whose data/ and features/ fill this bundle (e.g. E014)')
    run.add_argument('--tag', default=None, help='run label (default: HHMMSS)')
    run.add_argument('--session', default=None)
    run.add_argument('--timeout', type=float, default=7200, help='per-command timeout seconds')
    run.add_argument('--poll', type=float, default=20, help='poll interval seconds')
    run.add_argument('--keep', action='store_true', help='leave the VM running')
    run.add_argument('--promote', action='store_true', help='copy artifacts into canonical logs/ results/')
    run.set_defaults(func=run_job)

    listing = sub.add_parser('status', help='list sessions and compute-unit usage')
    listing.set_defaults(func=lambda _: print(call('colab', 'sessions') + call('colab', 'usage')))

    stop = sub.add_parser('stop', help='stop a session')
    stop.add_argument('--session', required=True)
    stop.set_defaults(func=lambda a: print(call('colab', 'stop', '-s', a.session)))

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
