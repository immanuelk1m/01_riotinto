"""Colab-side torch and snapshot smoke test before the long E017 run."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


base = Path("/content/E017")
sys.path.insert(0, str(base))
import E017 as experiment  # noqa: E402


expected = {
    "wti_crude_oil": 31,
    "copper": 30,
    "gold": 24,
    "corn": 35,
    "wheat": 30,
    "coffee": 20,
}
manifest = json.loads((base / "features" / "sheet_manifest.json").read_text())
for commodity, count in expected.items():
    names = [item["name"] for item in json.loads(
        (base / "features" / f"{commodity}_features.json").read_text()
    )["features"]]
    frame = pd.read_parquet(base / "data" / f"{commodity}_X.parquet")
    assert len(names) == count == manifest["tabs"][commodity]["feature_count"]
    assert names == list(frame.columns[1:])

torch.set_num_threads(2)
rng = np.random.default_rng(42)
features = rng.normal(size=(300, 8))
for window in (0, 52, 104):
    mapping = experiment.fit_mapping(features, 180, 220, window)
    changed = features.copy()
    changed[221:] = 1e12
    later_mapping = experiment.fit_mapping(changed, 180, 220, window)
    for left, right in zip(mapping, later_mapping):
        assert (left is None) == (right is None)
        if left is not None:
            assert all(np.array_equal(a, b) for a, b in zip(left, right))
    output = experiment.apply_mapping(features, mapping)
    assert np.isfinite(output).all()
    assert np.array_equal(output[:, -2:], features[:, -2:])
x = rng.normal(size=(96, experiment.core.SEQ, 8)).astype("float32")
y = rng.normal(size=(96, experiment.core.H)).astype("float32")
old_delta = experiment.core.MIN_DELTA
experiment.core.MIN_DELTA = 1e6
try:
    for direction in experiment.core.DIRECTIONS:
        experiment.configure(direction)
        model, epoch, history = experiment.core.fit_model(torch, x, y, "cpu", 2, (x[:24], y[:24]))
        assert epoch == 0 and len(history) == 3
        assert np.isfinite(experiment.core.predict(torch, model, x[:2], "cpu")).all()
finally:
    experiment.core.MIN_DELTA = old_delta
print(f"PASS torch={torch.__version__} cuda={torch.cuda.is_available()} snapshot/three-direction smoke")

