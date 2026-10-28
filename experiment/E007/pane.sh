#!/usr/bin/env bash
# E007 tmux pane: run one direction on Colab VM e007 -> poll every 60s -> download -> leaderboard.
# usage: pane.sh <purged|robust|anomaly|shallow|tsfm>
D=$1; S=${S:-e007}
cd "$(dirname "$0")"
sub() { sed "s/__DIR__/$D/g" "remote/$1"; }
sub launch.py | colab exec -s "$S"
while :; do
  out=$(sub poll.py | colab exec -s "$S" 2>&1)
  clear; echo "[E007_$D] $(date '+%H:%M:%S')"; echo "$out"
  grep -q "finished=True" <<<"$out" && break
  sleep 60
done
sub list.py | colab exec -s "$S" | sed 's/\x1b\[[0-9;]*m//g' | grep -o '/content/E007/[^ ]*' | while read -r f; do
  r=${f#/content/E007/}; mkdir -p "$(dirname "$r")"
  colab download -s "$S" "$f" "$r" >/dev/null 2>&1 || echo "FAIL $f"
done
cd ../.. && uv run experiment/E007/E007.py leaderboard "$D" 2>&1 | grep -E "E007_$D +h=all|transfer"
echo "PANE DONE E007_$D"
