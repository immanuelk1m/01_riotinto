#!/usr/bin/env python3
"""Fail the build when a number that can change a result enters the pipeline.

Three rules, all mechanical:

1. **No knob-carrying estimator.**  Anything in ``ESTIMATORS`` (Lasso, Ridge, the
   boosting/forest/neural constructors, PCA, ARIMA, …) carries free parameters, so
   it may not be constructed anywhere in the research code.  The retained models
   are the closed-form OLS/ADL and the naive forecast (`persistence`), both in ``_exo/engine.py``.
2. **No unregistered constant.**  A module-level ``NAME = <number>`` outside
   ``_exo/params.py`` is a research degree of freedom with no recorded origin.
3. **No literal in a knob position.**  A numeric literal handed to a call that
   changes the fitted object (``alpha=``, ``k=``, ``clip(±0.5)``, ``limit=8`` …) or
   a hard-coded calendar date used as a data filter.

Allowed without registration: small structural literals inside ordinary code
(``range(1, …)``, ``[:, 0]``, ``round(x, 6)``) — the rules above never look at
those, because they cannot change what is estimated.

Usage::

    uv run python tools/params_lint.py            # exit 1 on any violation
    uv run python tools/params_lint.py --json     # machine-readable report
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = (ROOT / "_exo", ROOT / "experiment", ROOT / "tools")

#: infrastructure and the linters themselves are out of scope
SKIP_PARTS = {"__pycache__", "colab_runs", "data", "results"}
SKIP_FILES = {"params.py", "params_lint.py", "params_inventory.py"}

#: estimator constructors carrying free parameters — the purge list
ESTIMATORS = {
    "Lasso", "LassoCV", "ElasticNet", "Ridge", "RidgeCV", "Lars", "OMP",
    "RandomForestRegressor", "ExtraTreesRegressor", "GradientBoostingRegressor",
    "HistGradientBoostingRegressor", "AdaBoostRegressor", "BaggingRegressor", "SVR",
    "MLPRegressor", "MLPClassifier", "PCA", "PLSRegression", "KernelPCA", "TruncatedSVD",
    "ARIMA", "SARIMAX", "ExponentialSmoothing", "VAR", "VARMAX", "AutoReg",
    "Sequential", "GRU", "LSTM", "Dense", "Conv1d", "TransformerEncoder",
    "LGBMRegressor", "XGBRegressor", "CatBoostRegressor",
    "GridSearchCV", "RandomizedSearchCV",
}

#: keyword arguments whose number decides the fitted object
KNOB_KWARGS = {
    "alpha", "lam", "lambda_", "k", "n_components", "n_estimators", "max_depth",
    "learning_rate", "min_samples_leaf", "l2_regularization", "hidden_layer_sizes",
    "min_periods", "limit", "min_rows", "max_iter", "tol", "window", "decay",
    "gamma", "shrink", "clip", "eps", "bandwidth", "dropout", "patience",
}

#: positional calls where a literal is a knob
KNOB_CALLS = {"clip", "_zscore", "_pick", "_ols", "_pcr3", "ffill", "rolling"}

DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: numbers that are structural in any position
STRUCTURAL = {0, 1, 2}


def files():
    for base in TARGETS:
        for path in sorted(base.rglob("*.py")):
            if SKIP_PARTS & set(path.parts) or path.name in SKIP_FILES:
                continue
            yield path


def _number(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(
        node.value, bool
    )


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return "?"


def check_file(path: Path) -> list[dict]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:  # a broken file is a lint failure of its own
        return [dict(file=str(path.relative_to(ROOT)), line=exc.lineno or 0, rule="syntax", detail=str(exc))]
    issues: list[dict] = []
    rel = str(path.relative_to(ROOT))

    #: a date literal is a data filter only when it is compared against data; the same
    #: literal starting a synthetic self-test frame is a fixture, not a degree of freedom
    compared_dates = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for operand in [node.left, *node.comparators]:
                if isinstance(operand, ast.Constant) and isinstance(operand.value, str):
                    if DATE.match(operand.value):
                        compared_dates.add(id(operand))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in ESTIMATORS:
                issues.append(
                    dict(file=rel, line=node.lineno, rule="estimator",
                         detail=f"{name}(...) carries free parameters and may not be constructed")
                )
            for keyword in node.keywords:
                if keyword.arg in KNOB_KWARGS and _number(keyword.value):
                    issues.append(
                        dict(file=rel, line=node.lineno, rule="knob-argument",
                             detail=f"{name}({keyword.arg}={keyword.value.value}) — register it in _exo/params.py")
                    )
            if name in KNOB_CALLS:
                for arg in node.args:
                    if _number(arg) and arg.value not in STRUCTURAL:
                        issues.append(
                            dict(file=rel, line=node.lineno, rule="knob-call",
                                 detail=f"{name}({arg.value}) — register it in _exo/params.py")
                        )
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id.isupper()
                    and _number(node.value)
                    and node.value.value not in STRUCTURAL
                ):
                    issues.append(
                        dict(file=rel, line=node.lineno, rule="unregistered-constant",
                             detail=f"{target.id} = {node.value.value} — move it to _exo/params.py")
                    )
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and DATE.match(node.value):
            if id(node) in compared_dates:
                issues.append(
                    dict(file=rel, line=node.lineno, rule="hardcoded-date",
                         detail=f"'{node.value}' as a data filter — resolve the series in config.json instead")
                )
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    issues: list[dict] = []
    count = 0
    for path in files():
        count += 1
        issues += check_file(path)

    if args.json:
        print(json.dumps(dict(files=count, issues=issues), indent=1))
    else:
        by_rule: dict[str, int] = {}
        for issue in issues:
            by_rule[issue["rule"]] = by_rule.get(issue["rule"], 0) + 1
        for issue in issues[:60]:
            print(f"{issue['file']}:{issue['line']}: [{issue['rule']}] {issue['detail']}")
        if len(issues) > 60:
            print(f"... {len(issues) - 60} more")
        print(f"\n[params_lint] {count} files, {len(issues)} violations {by_rule}")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
