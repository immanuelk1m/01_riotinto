#!/usr/bin/env bash
set -euo pipefail
S=${S:-e013-inv01}
DIRECTION=${DIRECTION:-inv01}
case "$DIRECTION" in erm|inv01|inv1) ;; *) exit 2 ;; esac
cd "$(dirname "$0")"
colab exec -s "$S" -f remote/launch.py
while :; do
  out=$(colab exec -s "$S" -f remote/poll.py)
  echo "$out"
  if grep -q 'failed=True' <<<"$out"; then exit 1; fi
  if grep -q 'finished=True' <<<"$out"; then break; fi
  sleep 30
done
fs=$(colab exec -s "$S" -f remote/list.py | sed 's/\x1b\[[0-9;]*m//g' | grep '^/content/E013/')
while IFS= read -r f; do
  r=${f#/content/E013/}
  mkdir -p "$(dirname "$r")"
  colab download -s "$S" "$f" "$r"
done <<<"$fs"
cd ../..
uv run experiment/E013/E013.py leaderboard "$DIRECTION"
