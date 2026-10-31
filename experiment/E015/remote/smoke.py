"""Run E015 snapshot and torch equivalence checks in the allocated runtime."""
import os
import subprocess
import sys
from pathlib import Path

base = Path("/content/E015")
subprocess.run([sys.executable, "test_e015.py", "--torch"], cwd=base, check=True,
               env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
import torch
if not torch.cuda.is_available():
    raise RuntimeError("E015 expects an allocated GPU")
print(f"PASS torch={torch.__version__} cuda={torch.cuda.get_device_name(0)}")
