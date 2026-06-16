"""マーケット指標・為替の取得（特徴量の素材）。

日経平均・TOPIX代理・ドル円・米国指数（前日終値）・VIX などを yfinance で取得する。
出力はロング形式: date, key, close
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from flowsignal import config
from flowsignal.data.prices import fetch_prices_yfinance


def fetch_market(
    start: str | dt.date, end: str | dt.date | None = None
) -> pd.DataFrame:
    """universe.yaml の market 定義をもとに指標の終値を取得。

    返り値カラム: date, key, close
    （key は "nikkei225" などの内部キー）
    """
    universe = config.load_universe()
    if not universe.market:
        return pd.DataFrame(columns=["date", "key", "close"])

    sym_to_key = {m.yf_symbol: m.key for m in universe.market}
    prices = fetch_prices_yfinance(universe.market_yf_symbols, start, end)
    if prices.empty:
        return pd.DataFrame(columns=["date", "key", "close"])

    out = prices[["date", "code", "close"]].copy()
    out["key"] = out["code"].map(sym_to_key)
    return out[["date", "key", "close"]]
