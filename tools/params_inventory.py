#!/usr/bin/env python3
"""Inventory every numeric literal in the research code, with its syntactic role.

The purge in ``docs/MAGIC_NUMBERS.md`` is only defensible if the list of things
being judged is complete and mechanically produced.  This tool is deliberately
dumb: it does not decide what is a magic number.  It records where each literal
appears and *how* it is used, because the two roles are judged differently:

``constant``    module-level ``NAME = 3``          -> may be a declared design constant
``call_arg``    ``Lasso(alpha=1e-3)`` / ``f(8)``   -> knob candidate: a free parameter
``default_arg`` ``def f(k: int = 3)``              -> knob candidate: a silent default
``compare``     ``if x > 0.5``                     -> usually code, sometimes a threshold
``inline``      ``x * 100.0``                      -> usually units, sometimes a fudge

Decisions live in ``docs/params_decisions.json`` (``id`` -> classification and
disposition) so that the judgement is reviewable in one place and the inventory
stays reproducible.

Usage::

    uv run python tools/params_inventory.py            # write docs/MAGIC_NUMBERS.{json,md}
    uv run python tools/params_inventory.py --check    # a literal has no decision yet
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = (ROOT / "_exo", ROOT / "experiment", ROOT / "tools")
SKIP_PARTS = {"__pycache__", "colab_runs", "data", "results", "node_modules"}
SKIP_FILES = {"params_inventory.py"}

#: the only literal kinds that can hide a research degree of freedom
KNOB_KINDS = ("constant", "call_arg", "default_arg")

OUT_JSON = ROOT / "docs" / "MAGIC_NUMBERS.json"
OUT_MD = ROOT / "docs" / "MAGIC_NUMBERS.md"
DECISIONS = ROOT / "docs" / "params_decisions.json"


def files():
    for base in TARGETS:
        for path in sorted(base.rglob("*.py")):
            if SKIP_PARTS & set(path.parts) or path.name in SKIP_FILES:
                continue
            yield path


def _is_number(node: ast.AST) -> bool:
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


def _keyword_of(call: ast.Call, node: ast.AST) -> str | None:
    for kw in call.keywords:
        if kw.value is node:
            return kw.arg or "**"
    return None


def scan(path: Path) -> list[dict]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()
    found: list[dict] = []

    def walk(node: ast.AST, parents: list[ast.AST], func: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(child, parents + [node], child.name)
                continue
            if _is_number(child):
                parent = node
                kind, call, argname = "inline", None, None
                if isinstance(parent, ast.Call) and any(child is a for a in parent.args):
                    kind, call = "call_arg", _call_name(parent)
                    index = [i for i, a in enumerate(parent.args) if a is child][0]
                    argname = f"#{index}"
                elif isinstance(parent, ast.Call):
                    kind, call, argname = "call_arg", _call_name(parent), _keyword_of(parent, child)
                elif isinstance(parent, ast.arguments):
                    kind = "default_arg"
                    argname = next(
                        (a.arg for a, d in zip(parent.args, parent.defaults) if d is child), None
                    )
                elif isinstance(parent, ast.Assign):
                    targets = [t for t in parent.targets if isinstance(t, ast.Name)]
                    kind = "constant" if targets else "inline"
                    argname = targets[0].id if targets else None
                elif isinstance(parent, ast.Compare):
                    kind = "compare"
                elif isinstance(parent, ast.AnnAssign) and isinstance(parent.target, ast.Name):
                    kind, argname = "constant", parent.target.id
                value = child.value
                found.append(
                    dict(
                        id=f"{path.relative_to(ROOT)}:{child.lineno}:{value}",
                        file=str(path.relative_to(ROOT)),
                        line=child.lineno,
                        col=child.col_offset,
                        value=value,
                        kind=kind,
                        func=func,
                        call=call,
                        argname=argname,
                        text=lines[child.lineno - 1].strip()[:110],
                    )
                )
            walk(child, parents + [node], func)

    walk(tree, [], None)
    return found


#: estimator constructors that carry their own free parameters
ESTIMATORS = {
    "Lasso", "LassoCV", "ElasticNet", "Ridge", "RidgeCV", "LinearRegression", "Lars", "OMP",
    "RandomForestRegressor", "ExtraTreesRegressor", "GradientBoostingRegressor",
    "HistGradientBoostingRegressor", "AdaBoostRegressor", "BaggingRegressor", "SVR",
    "MLPRegressor", "MLPClassifier", "PCA", "PLSRegression", "KernelPCA", "TruncatedSVD",
    "ARIMA", "SARIMAX", "ExponentialSmoothing", "VAR", "VARMAX", "AutoReg",
    "Sequential", "GRU", "LSTM", "Dense", "Conv1d", "TransformerEncoder",
    "LGBMRegressor", "XGBRegressor", "CatBoostRegressor",
}
#: function calls that reveal a search/selection loop rather than a fixed estimator
SEARCH_HINTS = {
    "array_split", "choice", "GENERATIONS", "predict_contrib", "permutation_importance",
    "cross_val_score", "GridSearchCV", "RandomizedSearchCV", "optuna", "study",
}


def fingerprint(path: Path) -> dict:
    """Estimator classes, search hints and module constants found in one file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    estimators, hints, constants = set(), set(), []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in ESTIMATORS:
                estimators.add(name)
            if name in SEARCH_HINTS:
                hints.add(name)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for target in targets:
                if isinstance(target, ast.Name) and target.id.isupper() and _is_number(value):
                    constants.append(f"{target.id}={value.value}")
    return dict(
        file=str(path.relative_to(ROOT)),
        estimators=sorted(estimators),
        search=sorted(hints),
        constants=constants,
    )


def load_decisions() -> dict:
    if not DECISIONS.exists():
        return {}
    return json.loads(DECISIONS.read_text(encoding="utf-8"))


def classify(rows: list[dict], decisions: dict) -> list[dict]:
    for row in rows:
        decision = decisions.get(row["id"], {})
        row["class"] = decision.get("class", "undecided")
        row["disposition"] = decision.get("disposition", "undecided")
        row["note"] = decision.get("note", "")
    return rows


def summary(rows: list[dict]) -> dict:
    by_kind: dict[str, int] = {}
    by_class: dict[str, int] = {}
    by_file: dict[str, dict[str, int]] = {}
    for row in rows:
        by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1
        by_class[row["class"]] = by_class.get(row["class"], 0) + 1
        slot = by_file.setdefault(row["file"], {})
        slot[row["kind"]] = slot.get(row["kind"], 0) + 1
    return dict(by_kind=by_kind, by_class=by_class, by_file=by_file)


def markdown(rows: list[dict], stats: dict, prints: list[dict]) -> str:
    knob_rows = [r for r in rows if r["kind"] in KNOB_KINDS]
    out = [
        "# Magic numbers and heuristics — inventory",
        "",
        "Generated by `tools/params_inventory.py`. Do not hand-edit: re-run the tool.",
        "Decisions are declared in [`params_decisions.json`](params_decisions.json).",
        "",
        "## Totals",
        "",
        f"- numeric literals scanned: **{len(rows)}**",
        f"- knob candidates (`constant` / `call_arg` / `default_arg`): **{len(knob_rows)}**",
        "",
        "| literal role | count |",
        "|---|---|",
    ]
    for kind, count in sorted(stats["by_kind"].items(), key=lambda kv: -kv[1]):
        out.append(f"| `{kind}` | {count} |")
    out += ["", "| classification | count |", "|---|---|"]
    for klass, count in sorted(stats["by_class"].items(), key=lambda kv: -kv[1]):
        out.append(f"| `{klass}` | {count} |")

    out += [
        "",
        "## Knob candidates",
        "",
        "Every literal whose value can change the fitted object.  `undecided` means the",
        "purge has not yet judged it, and the run must not proceed past Gate 2.",
        "",
        "| file:line | value | role | site | class | disposition |",
        "|---|---|---|---|---|---|",
    ]
    for row in sorted(knob_rows, key=lambda r: (r["file"], r["line"])):
        site = f"`{row['call']}({row['argname']})`" if row["call"] else f"`{row['argname'] or ''}`"
        out.append(
            f"| `{row['file']}:{row['line']}` | `{row['value']}` | {row['kind']} | {site} "
            f"| {row['class']} | {row['disposition']} |"
        )

    out += ["", "## Per file", "", "| file | constant | call_arg | default_arg | other |", "|---|---|---|---|---|"]
    for name, slot in sorted(stats["by_file"].items()):
        other = sum(v for k, v in slot.items() if k not in KNOB_KINDS)
        out.append(
            f"| `{name}` | {slot.get('constant', 0)} | {slot.get('call_arg', 0)} "
            f"| {slot.get('default_arg', 0)} | {other} |"
        )

    out += [
        "",
        "## Experiment model fingerprints",
        "",
        "What each experiment's own module actually fits.  A model is retired by the rule in",
        "the purge section when it carries a free parameter: an estimator constructor, a",
        "search loop, or a module constant that changes the fitted object.",
        "",
        "| experiment | estimators | search / selection | module constants |",
        "|---|---|---|---|",
    ]
    for row in prints:
        estimators = ", ".join(row["estimators"]) or "—"
        search = ", ".join(row["search"]) or "—"
        constants = ", ".join(row["constants"]) or "—"
        out.append(f"| `{row['experiment']}` | {estimators} | {search} | {constants} |")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if a knob candidate is undecided")
    args = parser.parse_args()

    rows: list[dict] = []
    for path in files():
        rows += scan(path)
    decisions = load_decisions()
    rows = classify(rows, decisions)
    stats = summary(rows)
    knob_rows = [r for r in rows if r["kind"] in KNOB_KINDS]

    prints = []
    for exp_dir in sorted((ROOT / "experiment").glob("E0*")):
        module = exp_dir / f"{exp_dir.name}.py"
        if not module.exists():
            continue
        item = fingerprint(module)
        item["experiment"] = exp_dir.name
        prints.append(item)

    (ROOT / "docs").mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(dict(rows=rows, summary=stats, fingerprints=prints), indent=1), encoding="utf-8")
    OUT_MD.write_text(markdown(rows, stats, prints), encoding="utf-8")

    undecided = [r for r in knob_rows if r["class"] == "undecided"]
    print(
        f"[inventory] {len(rows)} literals, {len(knob_rows)} knob candidates, "
        f"{len(undecided)} undecided -> docs/MAGIC_NUMBERS.md"
    )
    if args.check and undecided:
        for row in undecided[:40]:
            print(f"  undecided {row['file']}:{row['line']} = {row['value']} ({row['kind']})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
