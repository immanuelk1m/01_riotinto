#!/bin/zsh
# E028 weekly forward issue (README 개정 3). launchd runs it Monday 09:00 KST.
# mini: prepare full -> nothing to do if the DB snapshot did not move -> Colab forward
# -> append the log -> commit only the forward log and the snapshot behind it.
set -euo pipefail
cd "${0:A:h}/../.."
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
day=$(date -u +%F)
echo "== $day $(date -u +%T)Z"

uv run python experiment/E028/E028.py prepare full
if git diff --quiet -- experiment/E028/data; then
  echo "DB snapshot unchanged since the last issue: nothing issued"
  exit 0
fi

tag="fwd-$day"
uv run python -m _colab.job run --exp E028 --cmd forward --include _exo --tag "$tag"
cp "experiment/E028/colab_runs/$tag/results/forward_$day.csv" experiment/E028/forward/
grep -h "qualifying issues" "experiment/E028/colab_runs/$tag/logs/"*.out || true

if [[ $(git branch --show-current) == main ]]; then
  echo "on main: forward files left uncommitted"
  exit 0
fi
paths=(experiment/E028/forward experiment/E028/data/full experiment/E028/data/manifest_full.json)
git add -- $paths
git commit -q -m "E028 forward issue $day" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- $paths
git log --oneline -1
