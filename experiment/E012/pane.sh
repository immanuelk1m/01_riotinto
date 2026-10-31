#!/usr/bin/env bash
set -euo pipefail
S=${S:-e012}
cd "$(dirname "$0")"
colab exec -s "$S" -f remote/launch.py
while :; do
  out=$(colab exec -s "$S" -f remote/poll.py)
  echo "$out"
  if grep -q 'failed=True' <<<"$out"; then exit 1; fi
  if grep -q 'finished=True' <<<"$out"; then break; fi
  sleep 30
done
fs=$(colab exec -s "$S" -f remote/list.py | sed 's/\x1b\[[0-9;]*m//g' | grep '^/content/E012/')
while IFS= read -r f; do
  r=${f#/content/E012/}
  mkdir -p "$(dirname "$r")"
  colab download -s "$S" "$f" "$r"
done <<<"$fs"
cd ../..
uv run experiment/E012/E012.py leaderboard
