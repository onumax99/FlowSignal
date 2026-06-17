"""build.py の統合テスト。

サブモジュールのリーク防止は各 test_*.py で担保済みなので、ここでは
**結合の整合**を確認する:
- スキーマ（特徴量 24 列＝テクニカル13＋マーケット7＋カレンダー4）。
- マーケット特徴量が同一日付の全銘柄へ正しくブロードキャストされる。
- ラベル NaN 行が既定で落ちる / drop_unlabeled=False で残る。
- カレンダー特徴量の値域。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flowsignal.features.build import (
    CALENDAR_FEATURES,
    FEATURE_COLUMNS,
    build_dataset,
)
from flowsignal.features.market import MARKET_FEATURES
from flowsignal.features.technical import TECHNICAL_FEATURES

_MARKET_KEYS = {
    "nikkei225": 28000.0,
    "topix_etf": 2000.0,
    "usdjpy": 150.0,
    "sp500": 4000.0,
    "nasdaq": 14000.0,
    "vix": 15.0,
}


def _prices(codes, n=50, start="2024-01-01", seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    frames = []
    for code in codes:
        closes = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.015, n))
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "code": code,
                    "open": closes,
                    "high": closes,
                    "low": closes,
                    "close": closes,
                    "volume": rng.integers(1000, 5000, n).astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True), dates


def _market(dates, seed=1):
    rng = np.random.default_rng(seed)
    frames = [
        pd.DataFrame(
            {
                "date": dates,
                "key": key,
                "close": base * np.cumprod(1.0 + rng.normal(0, 0.01, len(dates))),
            }
        )
        for key, base in _MARKET_KEYS.items()
    ]
    return pd.concat(frames, ignore_index=True)


def test_feature_columns_count():
    assert len(FEATURE_COLUMNS) == len(TECHNICAL_FEATURES) + len(MARKET_FEATURES) + len(
        CALENDAR_FEATURES
    )
    assert len(FEATURE_COLUMNS) == 24


def test_schema():
    prices, dates = _prices(["AAA", "BBB"])
    df = build_dataset(prices, _market(dates))
    assert list(df.columns) == [
        "date",
        "code",
        *FEATURE_COLUMNS,
        "ret_fwd",
        "threshold",
        "label",
    ]


def test_no_unlabeled_rows_by_default():
    prices, dates = _prices(["AAA", "BBB"])
    df = build_dataset(prices, _market(dates))
    assert df["label"].notna().all()


def test_drop_unlabeled_false_keeps_more_rows():
    prices, dates = _prices(["AAA", "BBB"])
    kept = build_dataset(prices, _market(dates), drop_unlabeled=False)
    dropped = build_dataset(prices, _market(dates), drop_unlabeled=True)
    assert len(kept) > len(dropped)
    assert kept["label"].isna().any()


def test_market_features_broadcast_across_codes():
    prices, dates = _prices(["AAA", "BBB"])
    df = build_dataset(prices, _market(dates))
    # 同一日付では、市場特徴量は全銘柄で同値のはず。
    a_date = df["date"].iloc[len(df) // 2]
    rows = df[df["date"] == a_date]
    assert rows["code"].nunique() == 2
    for col in MARKET_FEATURES:
        assert rows[col].nunique(dropna=False) == 1


def test_calendar_feature_ranges():
    prices, dates = _prices(["AAA", "BBB"])
    df = build_dataset(prices, _market(dates))
    assert df["dow"].between(0, 4).all()
    assert df["month"].between(1, 12).all()
    assert set(df["is_month_start"].unique()).issubset({0, 1})
    assert set(df["is_month_end"].unique()).issubset({0, 1})
    # 50 営業日あれば月初は最低 1 つ含まれる。
    assert df["is_month_start"].sum() >= 1
