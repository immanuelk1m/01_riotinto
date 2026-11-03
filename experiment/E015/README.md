# E015: gate-to-deployment model consistency

E015 isolates the model/scaler transition between internal validation and the
next 26-week forecast block. All directions use the existing frozen E014 sheet
snapshot, six commodities, GRU/AdamW settings, seed 42 and zero risk penalty.

| Direction | Deployed model | Deployed scaler |
| --- | --- | --- |
| fresh | New model trained on all mature rows, exactly as E014 ERM | Refit on mature rows |
| frozen | Restored early-stop checkpoint used by the gate | Identical gate scaler |
| warm | Gate checkpoint continued on all mature rows | Identical gate scaler |

Warm resets AdamW moments because the E013 checkpoint stores model weights,
not optimizer state; the learning rate and weight decay remain unchanged.
Fresh and warm train for max(best_epoch, 1) additional epochs. Frozen adds no
training epochs. Gate kappa and deploy decisions are held fixed within each
block for all directions. Epoch-zero gate fallback remains persistence.
Curve artifacts record gate/deployed model and scaler SHA256 hashes.

Run locally: `uv run python experiment/E015/test_e015.py`.
Torch equivalence, frozen identity and warm-update tests run on Colab before
training. Entry point: `python E015.py fresh|frozen|warm [commodity ...]`.
Each direction has a separate Colab session and tmux pane invoked with
`S=e015-fresh DIRECTION=fresh bash experiment/E015/pane.sh`.
The leader allocates tmux/Colab; this implementation does not start them.

Data are read from E014 locally and uploaded without rebuilding from live DB.
Known E014 forced lag-zero and revised-vintage limitations remain. Prior
outer results have been inspected, so this is a retrospective controlled
ablation, not an untouched future holdout. No direction is selected on outer
loss. The fresh direction must match E014 ERM before interpreting contrasts.
Changing the model age in frozen and optimizer reset in warm are explicit
parts of this ablation.

Results retain the prediction and transfer schemas used by E013/E014. A
namespaced E015_naive baseline avoids overwriting the shared naive rows.
The launcher preserves a VM if downloads or local validation fail; it releases
the VM only after complete artifacts pass local validation.

## Retired 2026-11-03

This experiment cannot run. Model: GRU pools with fresh/frozen/warm starts. Removed constants: GRU; start-up schedule. Reason: the start-up schedule and the network are free parameters. Implementation preserved at commit `2ca48e9`; rows kept in the leaderboard under `status = 'retired-heuristic'`. See [`../RETIRED.md`](../RETIRED.md).
