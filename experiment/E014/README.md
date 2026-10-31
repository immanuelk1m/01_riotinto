# E014 sheet-defined feature-pool invariant GRU

This experiment reruns E013's fixed `erm`, `inv01`, and `inv1` directions with
the six commodity-specific feature pools from Google Sheet
`17mSdLi4RUOboC-ycC_QB4m0JhBBRedbU`.

`prepare.py` reuses release-safe `ml.<commodity>` columns when an approved
`ml.feature.effective_lag_weeks` mapping exists. At the user's explicit request,
the remaining sheet variables are read from `public.value`, resampled to
`W-SUN`, and used with zero extra publication lag. These columns are marked
`unapproved_timing=true` in `features/sheet_manifest.json`; E014 is therefore a
diagnostic comparison, not point-in-time causal evidence.

The GRU, AdamW optimizer, split/purge/gate/refit rules, and fixed penalties are
unchanged from E013. Wheat is added as the sixth tab and is reported separately
from the five-commodity comparison shared with E013.

Local preparation and checks:

```bash
uv run experiment/E014/prepare.py
uv run python experiment/E014/test_e014.py --torch
```

Remote execution uses one tmux session with one pane per direction:

```bash
tmux new-session -d -s E014-sheet-pool -n runs
tmux split-window -t E014-sheet-pool:0 -h
tmux split-window -t E014-sheet-pool:0.0 -v
tmux select-layout -t E014-sheet-pool:0 tiled
tmux send-keys -t E014-sheet-pool:0.0 'S=e014-erm DIRECTION=erm bash experiment/E014/pane.sh' Enter
tmux send-keys -t E014-sheet-pool:0.1 'S=e014-inv01 DIRECTION=inv01 bash experiment/E014/pane.sh' Enter
tmux send-keys -t E014-sheet-pool:0.2 'S=e014-inv1 DIRECTION=inv1 bash experiment/E014/pane.sh' Enter
```
