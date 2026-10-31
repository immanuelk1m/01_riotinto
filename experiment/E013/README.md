# E013 temporal-environment GRU ablation

Fixed directions: `erm` (lambda 0), `inv01` (0.1), `inv1` (1.0).
All use three contiguous equal-count environments and identical stratified
mini-batch sampling, so `erm` is the sampling-matched control. Report all
directions; choosing the best outer result does not establish future transfer.
This risk-variance ablation is inspired by invariant learning, not full FOIL.

Upload E013.py plus E008's five-commodity data/ and features/ into /content/E013.
For each independent Colab session upload run_config.json containing
`{"direction":"inv01"}` (or erm/inv1). Then run the corresponding tmux pane:
`S=e013-inv01 DIRECTION=inv01 bash experiment/E013/pane.sh`.
The leader owns session provisioning and stopping the Colab VM after download.

E010's 52-week early stop, 8-week purge, 52-week gate, 26-week refit, epoch-zero
fallback and five-commodity origin/horizon contract are retained. The gate
statistic remains diagnostic because kappa is fitted on that gate period.
Preselected E008 MI features retain their existing internal-validation bias;
this experiment does not claim to remove that limitation or verify data vintages.

Training curves include pooled loss, objective, environment risks and variance.
Transfer CSV and JSON report gate-to-next-block behavior retrospectively only.
Local test: `uv run python experiment/E013/test_e013.py`.
Torch objective/epoch-zero smoke: `python test_e013.py --torch` on Colab.
