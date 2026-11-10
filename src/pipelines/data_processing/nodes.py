"""preprocessing → 평가 격자. 격자의 y_h 는 로그수익률 `log(P_{t+h} / P_t)`(모델이 학습할 공간은 실험의
`target` 이 modeling 에서 정한다), 주는 `W-SUN`(일요일 마감)이다."""
import numpy as np
import pandas as pd

from src.utils import KEYS


def to_weekly(raw_prices: pd.DataFrame) -> pd.DataFrame:
    """주의 마지막 관측값. 결측·비양수(로그 불가)는 버린다."""
    weekly = (
        raw_prices.set_index("dt").groupby("series")["value"].resample("W-SUN").last()
        .dropna().rename_axis(KEYS).reset_index()
    )
    return weekly[weekly.value > 0].reset_index(drop=True)


def make_targets(weekly_prices: pd.DataFrame, horizon: int, origin_start: str) -> pd.DataFrame:
    """모든 실험이 공유하는 평가 격자: 시리즈별 달력 주 위의 `logp`, 타겟 y1..yH, origin 표시.

    격자를 달력으로 채우므로(`asfreq`) 빠진 주는 NaN 으로 남아 그 셀이 탈락할 뿐, 시점이 밀리지 않는다.
    """
    frames = []
    for series, group in weekly_prices.groupby("series"):
        logp = np.log(group.set_index("week").value.asfreq("W-SUN"))
        out = pd.DataFrame({"series": series, "logp": logp})
        for h in range(1, horizon + 1):
            out[f"y{h}"] = logp.shift(-h) - logp               # log(P_{t+h} / P_t)
        # 일요일의 day <= 7 이면 그 달의 첫 W-SUN 주
        out["is_origin"] = (out.index.day <= 7) & (out.index >= pd.Timestamp(origin_start))
        frames.append(out.reset_index())
    return pd.concat(frames, ignore_index=True)
