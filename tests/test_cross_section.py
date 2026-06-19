"""クロスセクション（相対）予測の特徴量変換・評価指標の単体テスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flowsignal.eval.cross_section_metrics import (
    bootstrap_mean_ci,
    daily_rank_ic,
    long_short_returns,
)
from flowsignal.features.cross_section import add_relative_target, cross_sectional_zscore


# --- 特徴量変換 --------------------------------------------------------------


def test_relative_target_is_demeaned_per_day():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "code": ["A", "B", "A", "B"],
            "ret_fwd": [0.01, 0.03, -0.02, 0.00],
        }
    )
    out = add_relative_target(df)
    # 各日デミーン: d1 mean=0.02 -> [-0.01,+0.01], d2 mean=-0.01 -> [-0.01,+0.01]
    assert out["rel_fwd"].round(6).tolist() == [-0.01, 0.01, -0.01, 0.01]
    # 各日合計はゼロ
    assert out.groupby("date")["rel_fwd"].sum().abs().max() == pytest.approx(0.0, abs=1e-12)


def test_relative_target_nan_when_single_name_day():
    df = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-01"]), "code": ["A"], "ret_fwd": [0.01]}
    )
    # 断面が 1 銘柄しかない日は相対化できない → NaN
    assert add_relative_target(df)["rel_fwd"].isna().all()


def test_cross_sectional_zscore_standardizes_and_zeros_constant():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 3),
            "code": ["A", "B", "C"],
            "f": [1.0, 2.0, 3.0],   # 断面で変動 → z-score
            "m": [5.0, 5.0, 5.0],   # 断面一定（市場/カレンダー相当）→ 0
        }
    )
    z = cross_sectional_zscore(df, ["f", "m"])
    assert z["f"].round(6).tolist() == [-1.0, 0.0, 1.0]  # std(ddof=1)=1
    assert z["m"].tolist() == [0.0, 0.0, 0.0]


def test_cross_sectional_zscore_preserves_warmup_nan():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 3),
            "code": ["A", "B", "C"],
            "f": [np.nan, 2.0, 4.0],
        }
    )
    z = cross_sectional_zscore(df, ["f"])
    assert np.isnan(z["f"].iloc[0])  # 入力 NaN は保持


# --- 評価指標 ----------------------------------------------------------------


def _panel(n_days=12, n_codes=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for c in range(n_codes):
            rows.append({"date": d, "code": c, "target": float(rng.normal())})
    return pd.DataFrame(rows)


def test_rank_ic_perfect_and_reversed():
    df = _panel()
    perfect = daily_rank_ic(df["target"], df["target"], df["date"], min_names=5)
    assert perfect["mean_ic"] == pytest.approx(1.0)
    assert perfect["n_days"] == 12
    reversed_ic = daily_rank_ic(-df["target"], df["target"], df["date"], min_names=5)
    assert reversed_ic["mean_ic"] == pytest.approx(-1.0)


def test_rank_ic_skips_thin_days():
    # 1 日 3 銘柄しかない → min_names=5 で除外され n_days=0
    df = _panel(n_days=4, n_codes=3)
    ic = daily_rank_ic(df["target"], df["target"], df["date"], min_names=5)
    assert ic["n_days"] == 0


def test_long_short_positive_when_score_matches_target():
    df = _panel()
    ls = long_short_returns(df["target"], df["target"], df["date"], quantile=0.25, min_names=5)
    assert ls["mean_spread"] > 0
    assert ls["mean_long"] > ls["mean_short"]


def test_bootstrap_mean_ci_brackets_mean():
    vals = np.arange(100) / 100.0
    lo, hi = bootstrap_mean_ci(vals, n_boot=300, seed=1)
    assert lo <= vals.mean() <= hi
