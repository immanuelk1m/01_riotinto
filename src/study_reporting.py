"""동일 예측 셀의 paired 비교와 월별 origin 블록 bootstrap. 파일·서비스 I/O는 없다."""

import numpy as np
import pandas as pd

CELL_KEYS = ["series", "origin", "horizon"]
PRICE_COLUMNS = ["fc", "actual", "base"]


def aligned_forecasts(candidate: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """키와 정답이 같은 두 예측을 정렬한다. 누락 셀의 교집합 비교는 허용하지 않는다."""
    frames = []
    for label, source in (("candidate", candidate), ("reference", reference)):
        missing = set(CELL_KEYS + PRICE_COLUMNS) - set(source.columns)
        if missing:
            raise ValueError(f"{label}: missing columns {sorted(missing)}")
        if source.empty:
            raise ValueError(f"{label}: empty forecasts")
        frame = source.copy()
        if frame[CELL_KEYS].isna().any(axis=None):
            raise ValueError(f"{label}: null cell keys")
        if not frame.series.map(lambda value: isinstance(value, str) and bool(value) and value != "__all__").all():
            raise ValueError(f"{label}: series must be nonempty strings other than '__all__'")
        frame["origin"] = pd.to_datetime(frame.origin, errors="raise")
        if frame.origin.isna().any():
            raise ValueError(f"{label}: null origins")
        horizons = pd.to_numeric(frame.horizon, errors="raise").to_numpy(dtype=float)
        if not (np.isfinite(horizons) & (horizons >= 1) & (horizons == np.floor(horizons))).all():
            raise ValueError(f"{label}: horizons must be positive integers")
        frame["horizon"] = horizons.astype(int)
        if frame.duplicated(CELL_KEYS).any():
            raise ValueError(f"{label}: duplicate cell keys")
        prices = frame[PRICE_COLUMNS].to_numpy(dtype=float)
        if not (np.isfinite(prices) & (prices > 0)).all():
            raise ValueError(f"{label}: fc, actual and base must be finite and positive")
        frames.append(frame.sort_values(CELL_KEYS).reset_index(drop=True))

    left, right = frames
    if not left[CELL_KEYS].equals(right[CELL_KEYS]):
        raise ValueError("candidate and reference have different cell keys")
    for column in ("actual", "base"):
        if not np.array_equal(left[column].to_numpy(), right[column].to_numpy()):
            raise ValueError(f"candidate and reference have different {column} values")
    return left, right


def _losses(candidate: pd.DataFrame, reference: pd.DataFrame, ape_cap: float) -> pd.DataFrame:
    if not np.isfinite(ape_cap) or ape_cap <= 0:
        raise ValueError("ape_cap must be finite and positive")
    left, right = aligned_forecasts(candidate, reference)
    ape = np.abs(left.fc.to_numpy(dtype=float) / left.actual.to_numpy(dtype=float) - 1)
    ref_ape = np.abs(right.fc.to_numpy(dtype=float) / right.actual.to_numpy(dtype=float) - 1)
    if not (np.isfinite(ape).all() and np.isfinite(ref_ape).all()):
        raise ValueError("forecast/actual ratio produced nonfinite loss")
    return left[CELL_KEYS].assign(
        loss=np.minimum(ape, ape_cap) * 100,
        reference_loss=np.minimum(ref_ape, ape_cap) * 100,
        uncapped_loss=ape * 100,
    )


def _gain(loss: float, reference_loss: float) -> float:
    # 상대 개선은 기준 오차가 0이면 정의되지 않는다(동일한 완벽 예측도 NaN).
    return float((1 - loss / reference_loss) * 100) if reference_loss > 0 else float("nan")


def _moving_block_indices(n_origins: int, samples: int, block_length: int,
                          rng: np.random.Generator) -> np.ndarray:
    """겹치는 연속 블록을 복원 추출한다. 모든 종목·horizon에 이 인덱스를 함께 적용한다."""
    n_blocks = (n_origins + block_length - 1) // block_length
    starts = rng.integers(0, n_origins - block_length + 1, size=(samples, n_blocks))
    return (starts[..., None] + np.arange(block_length)).reshape(samples, -1)[:, :n_origins]


def _scope_masks(columns: pd.MultiIndex):
    series = columns.get_level_values("series")
    horizons = columns.get_level_values("horizon")
    yield "overall", "__all__", 0, np.ones(len(columns), dtype=bool)
    for name in series.unique():
        yield "series", name, 0, np.asarray(series == name)
    for name, horizon in columns:
        yield "horizon", name, int(horizon), np.asarray((series == name) & (horizons == horizon))


def compare_forecasts(candidate: pd.DataFrame, reference: pd.DataFrame, ape_cap: float = 1.0,
                      bootstrap_samples: int = 2000, block_lengths: tuple[int, ...] = (3, 2, 4),
                      seed: int = 0) -> pd.DataFrame:
    """동일 셀의 전체·종목·종목×horizon 비교. 각 scope당 block 길이별 한 행.

    MAPE, delta_loss_pct와 CI는 percentage points, gain_pct는 상대 %다.
    CI는 paired loss(candidate-reference)의 pointwise 95% percentile 구간이다.
    bootstrap_pvalue는 평균 손실 차이 < 0에 대한 중심화 블록 bootstrap 단측 p값이다.
    전체 집계는 series×horizon 동일 가중치이며 모든 origin에서 완전한 패널을 요구한다.
    reference MAPE가 0이면 gain_pct=NaN. 너무 긴 블록은 n_origins로 줄여 기록한다.
    """
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, (int, np.integer)) or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    if not block_lengths or any(isinstance(length, bool) or not isinstance(length, (int, np.integer)) or length < 1
                                for length in block_lengths):
        raise ValueError("block_lengths must contain positive integers")
    losses = _losses(candidate, reference, ape_cap)
    panel = losses.pivot(index="origin", columns=["series", "horizon"], values="loss").sort_index().sort_index(axis=1)
    reference_panel = losses.pivot(index="origin", columns=["series", "horizon"], values="reference_loss").reindex_like(panel)
    expected_columns = pd.MultiIndex.from_product(
        [panel.columns.get_level_values("series").unique(), panel.columns.get_level_values("horizon").unique()],
        names=["series", "horizon"],
    )
    if panel.isna().any(axis=None) or not panel.columns.equals(expected_columns):
        raise ValueError("incomplete origin panel: every origin must contain all series/horizon cells")
    months = panel.index.to_period("M")
    if not months.is_unique or len(months) != len(pd.period_range(months.min(), months.max(), freq="M")):
        raise ValueError("origin blocks require exactly one origin in every consecutive calendar month")
    uncapped = losses.pivot(index="origin", columns=["series", "horizon"], values="uncapped_loss").reindex_like(panel)
    scopes = list(_scope_masks(panel.columns))
    weights = np.column_stack([mask / mask.sum() for _, _, _, mask in scopes])
    candidate_values, reference_values = panel.to_numpy(), reference_panel.to_numpy()
    origin_deltas = (candidate_values - reference_values) @ weights
    deltas = origin_deltas.mean(axis=0)
    candidate_means = candidate_values.mean(axis=0) @ weights
    reference_means = reference_values.mean(axis=0) @ weights
    n_origins = len(panel)
    rows = []
    # Clipping can make several requested lengths identical; emit each effective length once.
    for length in dict.fromkeys(min(int(length), n_origins) for length in block_lengths):
        indices = _moving_block_indices(n_origins, bootstrap_samples, length, np.random.default_rng(seed))
        counts = np.zeros((bootstrap_samples, n_origins), dtype=float)
        np.add.at(counts, (np.arange(bootstrap_samples)[:, None], indices), 1)
        bootstrap_means = counts @ origin_deltas / n_origins
        lower, upper = np.quantile(bootstrap_means, [0.025, 0.975], axis=0)
        null_means = bootstrap_means - deltas
        pvalues = (1 + (null_means <= deltas).sum(axis=0)) / (bootstrap_samples + 1)
        for i, (scope, name, horizon, mask) in enumerate(scopes):
            rows.append({
                "scope": scope, "series": name, "horizon": horizon,
                "mape": float(candidate_means[i]), "reference_mape": float(reference_means[i]),
                "gain_pct": _gain(candidate_means[i], reference_means[i]),
                "delta_loss_pct": float(deltas[i]), "ci_low": float(lower[i]), "ci_high": float(upper[i]),
                "block_length": length, "n_origins": n_origins, "n_cells": int(n_origins * mask.sum()),
                "bootstrap_pvalue": float(pvalues[i]), "max_ape_pct": float(uncapped.to_numpy()[:, mask].max()),
            })
    return pd.DataFrame(rows)


def annual_comparison(candidate: pd.DataFrame, reference: pd.DataFrame, ape_cap: float = 1.0) -> pd.DataFrame:
    """종목별·전체 연도별 비교. 전체 MAPE는 series×horizon 평균의 평균이다."""
    losses = _losses(candidate, reference, ape_cap)
    losses["year"] = losses.origin.dt.year
    rows = []
    for year, yearly in losses.groupby("year", sort=True):
        groups = [("overall", "__all__", yearly)]
        groups.extend(("series", name, group) for name, group in yearly.groupby("series", sort=True))
        for scope, name, group in groups:
            by_cell = group.groupby(["series", "horizon"])[["loss", "reference_loss"]].mean()
            loss, ref_loss = by_cell.mean().to_numpy()
            rows.append({
                "scope": scope, "series": name, "year": int(year), "mape": float(loss),
                "reference_mape": float(ref_loss), "gain_pct": _gain(loss, ref_loss),
                "delta_loss_pct": float(loss - ref_loss), "n_origins": int(group.origin.nunique()),
                "n_cells": len(group), "max_ape_pct": float(group.uncapped_loss.max()),
            })
    return pd.DataFrame(rows)


def holm_adjust(pvalues) -> np.ndarray:
    """Holm family-wise 보정. NaN은 결과에 보존하되 family 크기에는 포함한다."""
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1:
        raise ValueError("pvalues must be one-dimensional")
    if np.isinf(values).any() or ((values < 0) | (values > 1)).any():
        raise ValueError("pvalues must be within [0, 1] or NaN")
    missing = np.isnan(values)
    order = np.argsort(np.where(missing, 1.0, values), kind="stable")
    ordered = np.where(missing, 1.0, values)[order]
    adjusted = np.minimum(1, np.maximum.accumulate(ordered * np.arange(len(values), 0, -1)))
    result = np.empty_like(values)
    result[order] = adjusted
    result[missing] = np.nan
    return result
