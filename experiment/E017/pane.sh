#!/usr/bin/env bash
set -euo pipefail
S=${S:?set Colab session name}
DIRECTION=${DIRECTION:?set identity, qmap52, or qmap104}
case "$DIRECTION" in identity|qmap52|qmap104) ;; *) exit 2 ;; esac
cd "$(dirname "$0")"

STOP_SESSION=0
cleanup() {
  if [[ "$STOP_SESSION" -eq 1 ]]; then
    colab stop -s "$S" >/dev/null 2>&1 || true
  else
    echo "Preserving $S because artifacts were not fully downloaded" >&2
  fi
}
trap cleanup EXIT

run_with_deadline() {
  local seconds=$1
  shift
  "$@" &
  local command_pid=$!
  (sleep "$seconds"; kill -TERM "$command_pid" >/dev/null 2>&1 || true) >/dev/null 2>&1 &
  local watchdog_pid=$!
  local rc=0
  wait "$command_pid" || rc=$?
  kill "$watchdog_pid" >/dev/null 2>&1 || true
  wait "$watchdog_pid" 2>/dev/null || true
  return "$rc"
}

if ! colab status | grep -q "^\[$S\] "; then
  for attempt in $(seq 1 240); do
    if colab new -s "$S" --gpu T4; then
      break
    fi
    if [[ "$attempt" -eq 240 ]]; then
      echo "Unable to allocate $S after $attempt attempts" >&2
      exit 1
    fi
    echo "Waiting for a Colab GPU slot for $S (attempt $attempt/240)"
    sleep 30
  done
fi
colab exec -s "$S" -f remote/bootstrap.py
colab upload -s "$S" E017.py /content/E017/E017.py
colab upload -s "$S" ../E013/E013.py /content/E017/e013_core.py
colab upload -s "$S" "configs/$DIRECTION.json" /content/E017/run_config.json
for path in ../E014/data/*.parquet ../E014/features/*.json; do
  relative=${path#../E014/}
  colab upload -s "$S" "$path" "/content/E017/$relative"
done
colab exec -s "$S" -f remote/smoke.py
colab exec -s "$S" -f remote/launch.py

poll_failures=0
while :; do
  if ! out=$(run_with_deadline 30 colab exec -s "$S" -f remote/poll.py); then
    poll_failures=$((poll_failures + 1))
    if [[ "$poll_failures" -ge 20 ]]; then
      echo "Polling $S failed $poll_failures consecutive times" >&2
      exit 1
    fi
    echo "Transient poll failure for $S ($poll_failures/20); retrying"
    sleep 10
    continue
  fi
  poll_failures=0
  echo "$out"
  if grep -q 'failed=True' <<<"$out"; then break; fi
  if grep -q 'finished=True' <<<"$out"; then break; fi
  sleep 30
done

for attempt in $(seq 1 20); do
  if listing=$(run_with_deadline 30 colab exec -s "$S" -f remote/list.py); then
    break
  fi
  if [[ "$attempt" -eq 20 ]]; then
    echo "Unable to list remote artifacts for $S" >&2
    exit 1
  fi
  sleep 10
done
files=$(sed 's/\x1b\[[0-9;]*m//g' <<<"$listing" | grep '^/content/E017/' || true)
if [[ -z "$files" ]]; then
  echo "Empty artifact listing for $S; preserving VM" >&2
  exit 1
fi
while IFS= read -r file; do
  [[ -z "$file" ]] && continue
  relative=${file#/content/E017/}
  mkdir -p "$(dirname "$relative")"
  for attempt in $(seq 1 20); do
    if run_with_deadline 60 colab download -s "$S" "$file" "$relative"; then
      break
    fi
    if [[ "$attempt" -eq 20 ]]; then
      echo "Unable to download $file from $S" >&2
      exit 1
    fi
    sleep 10
  done
done <<<"$files"
if grep -q 'failed=True' <<<"$out"; then
  exit 1
fi
cd ../..
uv run experiment/E017/E017.py leaderboard "$DIRECTION"
STOP_SESSION=1

