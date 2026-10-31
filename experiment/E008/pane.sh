#!/usr/bin/env bash
# E008 tmux pane: run one direction on Colab VM e008 -> poll every 60s -> download -> leaderboard.
# usage: pane.sh <global|dro|avg|huber|anytime>
D=$1; S=${S:-e008}
cd "$(dirname "$0")"
sub() { sed "s/__DIR__/$D/g" "remote/$1"; }
sub launch.py | colab exec -s "$S"
while :; do
  out=$(sub poll.py | colab exec -s "$S" 2>&1)
  clear; echo "[E008_$D] $(date '+%H:%M:%S')"; echo "$out"
  grep -q "finished=True" <<<"$out" && break
  sleep 60
done
# colab exec sometimes hangs (killed by the watcher) -> retry until the listing is non-empty
until fs=$(sub list.py | colab exec -s "$S" | sed 's/\x1b\[[0-9;]*m//g' | grep -o '/content/E008/[^ ]*'); do sleep 10; done
for f in $fs; do
  r=${f#/content/E008/}; mkdir -p "$(dirname "$r")"
  colab download -s "$S" "$f" "$r" >/dev/null 2>&1 || echo "FAIL $f"
done
cd ../.. && uv run experiment/E008/E008.py leaderboard "$D" 2>&1 | grep -E "E008_$D +h=all|transfer"
echo "PANE DONE E008_$D"
