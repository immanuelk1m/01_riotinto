#!/usr/bin/env bash
set -euo pipefail
S=${S:?set Colab session name}
DIRECTION=${DIRECTION:?set fresh, frozen, or warm}
case "$DIRECTION" in fresh|frozen|warm) ;; *) exit 2 ;; esac
cd "$(dirname "$0")"
mkdir -p logs
STOP_SESSION=0
cleanup() {
  if [[ "$STOP_SESSION" -eq 1 ]]; then
    colab stop -s "$S"
  else
    echo "Preserving $S: artifacts have not passed local validation" >&2
  fi
}
trap cleanup EXIT

run_with_deadline() {
  python3 -c '
import os, signal, subprocess, sys
p = subprocess.Popen(sys.argv[2:], start_new_session=True)
try:
    rc = p.wait(timeout=float(sys.argv[1]))
except subprocess.TimeoutExpired:
    os.killpg(p.pid, signal.SIGKILL)
    p.wait()
    rc = 124
sys.exit(rc)
' "$@"
}

retry() {
  local attempt
  for attempt in $(seq 1 6); do
    if run_with_deadline 90 "$@"; then return 0; fi
    if [[ "$attempt" -lt 6 ]]; then sleep 10; fi
  done
  return 1
}

if [[ "${RESUME:-0}" != 1 ]]; then
  status=$(colab status)
  if ! grep -Fq "[$S] " <<<"$status"; then
    for attempt in $(seq 1 120); do
      if run_with_deadline 90 colab new -s "$S" --gpu T4; then break; fi
      if [[ "$attempt" -eq 120 ]]; then exit 1; fi
      echo "Waiting for Colab GPU slot: $S"
      sleep 30
    done
  fi
  retry colab exec -s "$S" -f remote/bootstrap.py
  retry colab upload -s "$S" E015.py /content/E015/E015.py
  retry colab upload -s "$S" test_e015.py /content/E015/test_e015.py
  retry colab upload -s "$S" ../E013/E013.py /content/E015/reference_e013.py
  retry colab upload -s "$S" "configs/$DIRECTION.json" /content/E015/run_config.json
  for path in ../E014/data/*.parquet ../E014/features/*.json; do
    relative=${path#../E014/}
    retry colab upload -s "$S" "$path" "/content/E015/$relative"
  done
  retry colab exec -s "$S" -f remote/smoke.py --timeout 90
  # Launch is non-idempotent. On ambiguous network failure, inspect then RESUME=1.
  run_with_deadline 90 colab exec -s "$S" -f remote/launch.py
fi

poll_failures=0
while :; do
  if ! out=$(run_with_deadline 45 colab exec -s "$S" -f remote/poll.py); then
    poll_failures=$((poll_failures + 1))
    if [[ "$poll_failures" -ge 20 ]]; then exit 1; fi
    sleep 10
    continue
  fi
  poll_failures=0
  echo "$out"
  if grep -q 'failed=True' <<<"$out"; then break; fi
  if grep -q 'finished=True' <<<"$out"; then break; fi
  sleep 30
done

listing=$(retry colab exec -s "$S" -f remote/list.py)
files=$(sed 's/\x1b\[[0-9;]*m//g' <<<"$listing" | grep '^/content/E015/')
while IFS= read -r file; do
  [[ -z "$file" ]] && continue
  relative=${file#/content/E015/}
  mkdir -p "$(dirname "$relative")"
  retry colab download -s "$S" "$file" "$relative"
done <<<"$files"

if grep -q 'failed=True' <<<"$out"; then exit 1; fi
[[ -f "logs/DONE_$DIRECTION" && ! -f "logs/FAILED_$DIRECTION" ]]
uv run python E015.py leaderboard "$DIRECTION"
STOP_SESSION=1
