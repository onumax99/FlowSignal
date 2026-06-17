"""指標・為替を特徴量化し、JP 取引日へ時点整合でマージする。

時点整合（M2 で最大のリーク源・規則をここで固定する）:
- 特徴量行の日付 t は「JP 取引日 t の大引け（15:00 JST）時点の状態」を表す。
- **JP セッション系**（nikkei225, topix_etf）: 当日 t の終値を使う（大引けに既知）。
- **米国セッション系**（sp500, nasdaq, vix）: US 取引日 t の終値が確定するのは
  翌朝（~6:00 JST t+1）で、JP 大引け t には未確定。よって row t には
  **t の前日（カレンダー）までに確定済みの直近米クローズ = t-1 の overnight** を割り当てる。
  → STATUS §6.2 の規則「JST 取引日 t に使う米指数 = t-1 の overnight」と一致。
- **usdjpy**: yfinance の日次終値は NY クローズ基準で JP 大引け後に確定するため、
  保守的に米国系と同様 1 日前（≤ t-1）へ整合する（leak-safe を優先）。
- 祝日 NaN は **過去方向のみの ffill** で補完（未来参照を入れない）。

整合は「カレンダー t-1 日以前の直近値」を引く方式（`reindex(t-1, ffill)`）で行うため、
JP 連休や週末を跨いでも常に「t の大引けで既知の値」だけを使う（過剰に保守的になることはあっても
未来を参照しない）。

ラベル（labels.py）は row t に対し t→t+1 の翌日リターンを付与する想定。

入出力:
    入力 : market（ロング: date, key, close）, trading_dates（JP 取引日の列）
    出力 : date 行 × MARKET_FEATURES（JP 取引日に整合済み・昇順）
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

# 出力されるマーケット特徴量の列名（build.py / eval から参照する正準リスト）。
MARKET_FEATURES: list[str] = [
    "ret_nikkei",
    "ret_topix",
    "ret_usdjpy",
    "ret_sp500",
    "ret_nasdaq",
    "chg_vix",
    "vix_level",
]

# JP 大引け t に既知（当日終値を使う）
_CONTEMP_KEYS = ("nikkei225", "topix_etf")
# JP 大引け t には未確定（≤ t-1 の直近値へ整合する）
_OVERNIGHT_KEYS = ("usdjpy", "sp500", "nasdaq", "vix")
_REQUIRED_KEYS = (*_CONTEMP_KEYS, *_OVERNIGHT_KEYS)


def compute_market_features(
    market: pd.DataFrame, trading_dates: Iterable
) -> pd.DataFrame:
    """マーケット指標を JP 取引日に時点整合した特徴量へ変換する。

    Args:
        market: date, key, close を含むロング形式 DataFrame。
        trading_dates: 整合先の JP 取引日（prices の date 列を想定）。

    Returns:
        date, MARKET_FEATURES を持つワイド形式 DataFrame（date 昇順）。
    """
    for col in ("date", "key", "close"):
        if col not in market.columns:
            raise ValueError(f"market に必要な列がありません: {col}")

    cal = pd.DatetimeIndex(
        sorted(pd.to_datetime(pd.Index(trading_dates)).unique())
    )
    if cal.empty:
        return pd.DataFrame(columns=["date", *MARKET_FEATURES])

    wide = (
        market.assign(date=pd.to_datetime(market["date"]))
        .pivot(index="date", columns="key", values="close")
        .sort_index()
    )
    missing = [k for k in _REQUIRED_KEYS if k not in wide.columns]
    if missing:
        raise ValueError(f"market に必要な key がありません: {missing}")

    # 祝日 NaN を過去方向のみ補完（未来参照を入れない）。
    wide = wide.ffill()

    # JP セッション系: t の値（≤ t の直近）。
    contemp = wide[list(_CONTEMP_KEYS)].reindex(cal, method="ffill")
    # 米国・FX 系: ≤ t-1（カレンダー前日以前の直近）の値へ整合し、index を t に振り直す。
    prev = cal - pd.Timedelta(days=1)
    overnight = wide[list(_OVERNIGHT_KEYS)].reindex(prev, method="ffill")
    overnight.index = cal

    w = pd.concat([contemp, overnight], axis=1)

    out = pd.DataFrame(index=cal)
    out["ret_nikkei"] = w["nikkei225"].pct_change(fill_method=None)
    out["ret_topix"] = w["topix_etf"].pct_change(fill_method=None)
    out["ret_usdjpy"] = w["usdjpy"].pct_change(fill_method=None)
    out["ret_sp500"] = w["sp500"].pct_change(fill_method=None)
    out["ret_nasdaq"] = w["nasdaq"].pct_change(fill_method=None)
    out["chg_vix"] = w["vix"].pct_change(fill_method=None)
    out["vix_level"] = w["vix"]  # 市場レジームの代理（水準そのもの）

    return out.reset_index(names="date")[["date", *MARKET_FEATURES]]
