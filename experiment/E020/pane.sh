#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec uv run python runner.py --session "${S:?set session}" --direction "${DIRECTION:?set direction}"
