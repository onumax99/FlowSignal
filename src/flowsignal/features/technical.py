"""テクニカル指標の自前実装（銘柄ごと・t 時点まで）。

リーク防止の方針（M2 の中核要件）:
- すべての指標は後方参照（diff / rolling / ewm）のみで計算し、各日付 t の値は
  **t 以前の終値・出来高だけ**に依存する（未来を参照しない）。この性質は
  単体テスト（test_technical.py）で「系列を未来側に切っても過去の値が変わらない」
  ことを assert して担保する。
- ウォームアップ期間（移動平均などの先頭）の NaN はそのまま返す。欠損補完や
  ラベルとの時点整合（features(t) → label(t→t+1)）は build.py 側で担う。
  lightgbm / HistGradientBoosting は NaN をネイティブに扱えるため埋めない。
- pandas-ta 等の外部ライブラリは使わない（計算を内製してリークを制御するため）。

入出力はロング形式で統一する:
    入力 : date, code, close, volume（open/high/low があっても無視）
    出力 : date, code, <TECHNICAL_FEATURES...>
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 出力されるテクニカル特徴量の列名（build.py / eval から参照する正準リスト）。
TECHNICAL_FEATURES: list[str] = [
    "ret_1",
    "ret_5",
    "ret_10",
    "sma5_dev",
    "sma10_dev",
    "sma20_dev",
    "rsi14",
    "macd",
    "macd_signal",
    "macd_hist",
    "hv20",
    "vol_chg",
    "vol_ratio20",
]

_REQUIRED_COLUMNS = ("date", "code", "close", "volume")


def _ema(s: pd.Series, span: int) -> pd.Series:
    """指数移動平均（adjust=False の再帰式・span 本そろうまでは NaN）。"""
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder 型 RSI（EWMA 平滑, alpha=1/period）。値域 [0, 100]。

    - 全勝区間（損失平均=0, 利得平均>0）は RSI=100。
    - 全敗区間（利得平均=0）は RSI=0。
    - 無変化区間（利得・損失とも 0）は判断不能のため NaN のまま。
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    # avg_loss=0 での 0 除算（inf/警告）を避けるため NaN 経由で計算し、
    # 全勝ケースだけ明示的に 100 を入れ直す。
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(~((avg_loss == 0.0) & (avg_gain > 0.0)), 100.0)
    return rsi


def _macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD・シグナル・ヒストグラム。"""
    macd = _ema(close, fast) - _ema(close, slow)
    macd_signal = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def _compute_one(g: pd.DataFrame) -> pd.DataFrame:
    """単一銘柄（日付昇順）のテクニカル特徴量を計算する。"""
    g = g.sort_values("date").reset_index(drop=True)
    close = g["close"].astype("float64")
    volume = g["volume"].astype("float64")
    macd, macd_signal, macd_hist = _macd(close)

    feats = {
        "date": g["date"],
        "code": g["code"],
        # リターン（fill_method=None で前方補完を無効化＝欠損は欠損のまま）
        "ret_1": close.pct_change(1, fill_method=None),
        "ret_5": close.pct_change(5, fill_method=None),
        "ret_10": close.pct_change(10, fill_method=None),
        # 移動平均乖離率 = close / SMA(n) - 1
        "sma5_dev": close / close.rolling(5, min_periods=5).mean() - 1.0,
        "sma10_dev": close / close.rolling(10, min_periods=10).mean() - 1.0,
        "sma20_dev": close / close.rolling(20, min_periods=20).mean() - 1.0,
        "rsi14": _rsi(close, 14),
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        # ヒストリカルボラ = 日次リターンの 20 日標準偏差
        "hv20": close.pct_change(fill_method=None).rolling(20, min_periods=20).std(),
        # 出来高変化（前日比）と 20 日平均出来高に対する比率
        "vol_chg": volume.pct_change(fill_method=None),
        "vol_ratio20": volume / volume.rolling(20, min_periods=20).mean() - 1.0,
    }
    return pd.DataFrame(feats)


def compute_technical(prices: pd.DataFrame) -> pd.DataFrame:
    """ロング形式の株価から銘柄ごとのテクニカル特徴量を計算する。

    Args:
        prices: date, code, close, volume を含むロング形式 DataFrame。

    Returns:
        date, code, <TECHNICAL_FEATURES...> を持つロング形式 DataFrame
        （code, date の昇順）。先頭のウォームアップ区間は NaN を含む。
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in prices.columns]
    if missing:
        raise ValueError(f"prices に必要な列がありません: {missing}")

    if prices.empty:
        return pd.DataFrame(columns=["date", "code", *TECHNICAL_FEATURES])

    frames = [_compute_one(g) for _, g in prices.groupby("code", sort=False)]
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["code", "date"]).reset_index(drop=True)
