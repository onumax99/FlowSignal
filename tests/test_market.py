"""market.py の単体テスト。

最重要は **時点整合（リーク防止）** の検証:
- JP セッション系（日経・TOPIX）は当日 t の値を使う。
- 米国・FX 系（S&P500/NASDAQ/VIX/USDJPY）は ≤ t-1 の overnight 値を使う
  （JP 大引け t には未確定の当日 US 終値を絶対に使わない）。
- 祝日 NaN は過去方向のみ ffill（未来参照しない）。
- 系列を未来側で切っても過去の特徴量が一致する（先読み無し）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flowsignal.features.market import MARKET_FEATURES, compute_market_features

# テストは「連続カレンダー日」を取引日に見立て、t-1 整合を 1 インデックスずれとして
# 厳密に検証できるようにする（freq="D"）。
_KEYS = ("nikkei225", "topix_etf", "usdjpy", "sp500", "nasdaq", "vix")


def _market(closes_by_key, start="2024-01-01"):
    n = len(next(iter(closes_by_key.values())))
    dates = pd.date_range(start, periods=n, freq="D")
    frames = [
        pd.DataFrame({"date": dates, "key": key, "close": list(closes)})
        for key, closes in closes_by_key.items()
    ]
    return pd.concat(frames, ignore_index=True), dates


def _standard():
    return {
        "nikkei225": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        "topix_etf": [200.0, 201.0, 202.0, 203.0, 204.0, 205.0],
        "usdjpy": [150.0, 151.0, 152.0, 153.0, 154.0, 155.0],
        "sp500": [4000.0, 4010.0, 4020.0, 4030.0, 4040.0, 4050.0],
        "nasdaq": [14000.0, 14010.0, 14020.0, 14030.0, 14040.0, 14050.0],
        "vix": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    }


def test_output_schema():
    mkt, dates = _market(_standard())
    out = compute_market_features(mkt, dates)
    assert list(out.columns) == ["date", *MARKET_FEATURES]
    assert len(out) == len(dates)


def test_missing_key_raises():
    mkt, dates = _market({k: [1.0, 2.0, 3.0] for k in _KEYS if k != "vix"})
    try:
        compute_market_features(mkt, dates)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_jp_series_contemporaneous():
    # 日経は当日 t の終値を使う → ret_nikkei[i] = nikkei[i]/nikkei[i-1]-1。
    mkt, dates = _market(_standard())
    out = compute_market_features(mkt, dates)
    np.testing.assert_allclose(
        out["ret_nikkei"].to_numpy(dtype=float),
        [np.nan, 101 / 100 - 1, 102 / 101 - 1, 103 / 102 - 1, 104 / 103 - 1, 105 / 104 - 1],
        equal_nan=True,
        rtol=1e-9,
    )


def test_us_series_use_overnight_t_minus_1():
    # VIX 水準は ≤ t-1 の値（当日 US 終値は未使用）。
    # vix_raw = [10,11,12,13,14,15] → vix_level = [NaN,10,11,12,13,14]
    mkt, dates = _market(_standard())
    out = compute_market_features(mkt, dates)
    np.testing.assert_allclose(
        out["vix_level"].to_numpy(dtype=float),
        [np.nan, 10.0, 11.0, 12.0, 13.0, 14.0],
        equal_nan=True,
    )
    # S&P500 リターンも overnight 同士の比 → ret_sp500[2] = sp500[1]/sp500[0]-1。
    ret_sp = out["ret_sp500"].to_numpy(dtype=float)
    assert np.isnan(ret_sp[0]) and np.isnan(ret_sp[1])
    np.testing.assert_allclose(ret_sp[2], 4010.0 / 4000.0 - 1, rtol=1e-9)


def test_holiday_nan_filled_from_past_only():
    # VIX に中抜け NaN を入れる → 過去方向 ffill で埋まり、未来値は使われない。
    closes = _standard()
    closes["vix"] = [10.0, 11.0, 12.0, np.nan, 14.0, 15.0]
    mkt, dates = _market(closes)
    out = compute_market_features(mkt, dates)
    # row4 の vix_level = ffill 後の vix[3] = 12（過去由来）であり、14 ではない。
    np.testing.assert_allclose(out["vix_level"].to_numpy(dtype=float)[4], 12.0)


def test_no_lookahead_truncation_invariance():
    rng = np.random.default_rng(7)
    n = 60
    closes = {
        k: (base * np.cumprod(1.0 + rng.normal(0, 0.01, n))).tolist()
        for k, base in zip(
            _KEYS, [100, 200, 150, 4000, 14000, 10], strict=True
        )
    }
    mkt, dates = _market(closes)

    full = compute_market_features(mkt, dates)
    cut_mkt = mkt[mkt["date"].isin(dates[:40])]
    cut = compute_market_features(cut_mkt, dates[:40])

    pd.testing.assert_frame_equal(
        full.iloc[:40].reset_index(drop=True),
        cut.reset_index(drop=True),
    )
