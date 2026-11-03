#!/bin/zsh
# One pane = one pre-registered direction. Orchestrated locally, executed on Colab.
set -u
export PATH="$HOME/.local/bin:$PATH"
exp="$1"; dir="$2"; wait_exp="${3:-}"; wait_n="${4:-0}"; slots="${5:-3}"
cd /Users/sonet/Desktop/01_riotinto || exit 1
if [ -n "$wait_exp" ] && [ "$wait_exp" != "-" ]; then
  while [ "$(find "experiment/$wait_exp/logs" -name 'pane-*.rc' 2>/dev/null | wc -l | tr -d ' ')" -lt "$wait_n" ]; do
    sleep 20
  done
fi
mkdir -p "experiment/$exp/logs"
# crude concurrency limiter: at most $slots Colab sessions at a time across panes
slot=""
while [ -z "$slot" ]; do
  for i in $(seq 1 "$slots"); do
    if mkdir "/tmp/exo_slot_$i" 2>/dev/null; then slot="/tmp/exo_slot_$i"; break; fi
  done
  [ -z "$slot" ] && sleep 15
done
uv run python -m _colab.job run --exp "$exp" --cmd "$dir" --tag "$dir" \
  --include _exo --timeout 7200 --poll 30 > "experiment/$exp/logs/pane-$dir.log" 2>&1
echo $? > "experiment/$exp/logs/pane-$dir.rc"
rmdir "$slot" 2>/dev/null
