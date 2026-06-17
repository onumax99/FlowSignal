"""学習用テーブルの生成（テクニカル＋マーケット＋カレンダー＋ラベルの結合）。

各サブモジュールが時点整合（リーク防止）を担保済みのため、ここでは **(date, code)
を鍵に素直に結合**するだけでよい:
- technical.py : 銘柄ごとのテクニカル特徴量（t 時点まで）
- market.py    : マーケット特徴量（JP 当日 / 米系は t-1 overnight に整合済み）
- labels.py    : 3 クラスラベル（row t = t→t+1 の翌日方向）
- カレンダー   : 曜日・月・月初/月末（決算期の季節性の代理。暦は事前確定なので非リーク）

出力は 1 行 = (date, code) の学習用ロングテーブル。特徴量のウォームアップ NaN は
そのまま残す（lightgbm がネイティブに扱う）。ラベル NaN 行（ウォームアップ・各銘柄
最終日）は既定で落とす。
"""

from __future__ import annotations

import pandas as pd

from flowsignal.features.labels import compute_labels
from flowsignal.features.market import MARKET_FEATURES, compute_market_features
from flowsignal.features.technical import TECHNICAL_FEATURES, compute_technical

CALENDAR_FEATURES: list[str] = ["dow", "month", "is_month_start", "is_month_end"]

# モデルに渡す説明変数の正準リスト（この順序で X を組む）。
FEATURE_COLUMNS: list[str] = [
    *TECHNICAL_FEATURES,
    *MARKET_FEATURES,
    *CALENDAR_FEATURES,
]


def _calendar_features(dates: pd.Series) -> pd.DataFrame:
    """ユニーク日付ごとのカレンダー特徴量（暦は事前確定＝非リーク）。"""
    u = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
    cal = pd.DataFrame({"date": u})
    cal["dow"] = u.dayofweek  # 0=月 .. 4=金
    cal["month"] = u.month
    by_month = cal.groupby(u.to_period("M"))["date"]
    cal["is_month_start"] = (cal["date"] == by_month.transform("min")).astype(int)
    cal["is_month_end"] = (cal["date"] == by_month.transform("max")).astype(int)
    return cal


def build_dataset(
    prices: pd.DataFrame,
    market: pd.DataFrame,
    *,
    label_mode: str = "vol",
    k: float = 0.5,
    fixed_threshold: float = 0.007,
    vol_window: int = 20,
    drop_unlabeled: bool = True,
) -> pd.DataFrame:
    """学習用テーブルを生成する。

    Args:
        prices: date, code, OHLCV のロング形式。
        market: date, key, close のロング形式。
        label_mode / k / fixed_threshold / vol_window: labels.compute_labels に渡す。
        drop_unlabeled: True なら label が NaN の行（ウォームアップ・各銘柄最終日）を落とす。

    Returns:
        date, code, <FEATURE_COLUMNS...>, ret_fwd, threshold, label を持つ DataFrame
        （code, date 昇順）。
    """
    tech = compute_technical(prices)
    mkt = compute_market_features(market, prices["date"])
    lab = compute_labels(
        prices,
        mode=label_mode,
        k=k,
        fixed_threshold=fixed_threshold,
        vol_window=vol_window,
    )
    cal = _calendar_features(prices["date"])

    df = (
        tech.merge(
            lab[["date", "code", "ret_fwd", "threshold", "label"]],
            on=["date", "code"],
            how="left",
        )
        .merge(mkt, on="date", how="left")
        .merge(cal, on="date", how="left")
    )

    if drop_unlabeled:
        df = df[df["label"].notna()].copy()

    ordered = ["date", "code", *FEATURE_COLUMNS, "ret_fwd", "threshold", "label"]
    return df[ordered].sort_values(["code", "date"]).reset_index(drop=True)


def load_and_build(**kwargs) -> pd.DataFrame:
    """data/raw の parquet を読み込んで build_dataset を実行する薄いラッパ。"""
    from flowsignal.data.storage import load_parquet

    prices = load_parquet("prices")
    market = load_parquet("market")
    return build_dataset(prices, market, **kwargs)
