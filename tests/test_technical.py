"""technical.py の単体テスト。

狙いは 2 点:
1. 既知値照合 — 単純な系列で指標値が定義どおりか。
2. **先読み無し（リーク制御）** — 系列を未来側で切っても過去の特徴量が
   1 ビットも変わらないこと。M2 のリーク防止の根拠になる最重要テスト。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flowsignal.features.technical import (
    TECHNICAL_FEATURES,
    compute_technical,
)


def _frame(closes, code="AAA", volumes=None, start="2024-01-01") -> pd.DataFrame:
    """テスト用のロング形式株価フレームを作る（営業日連番）。"""
    n = len(closes)
    dates = pd.bdate_range(start, periods=n)
    vols = volumes if volumes is not None else [1000.0] * n
    return pd.DataFrame(
        {
            "date": dates,
            "code": code,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": vols,
        }
    )


def test_output_schema():
    out = compute_technical(_frame([100.0] * 30))
    assert list(out.columns) == ["date", "code", *TECHNICAL_FEATURES]
    assert len(out) == 30


def test_required_columns_validated():
    with pytest.raises(ValueError):
        compute_technical(pd.DataFrame({"date": [], "code": []}))


def test_empty_input_returns_empty_schema():
    out = compute_technical(
        pd.DataFrame(columns=["date", "code", "close", "volume"])
    )
    assert out.empty
    assert list(out.columns) == ["date", "code", *TECHNICAL_FEATURES]


def test_returns_known_values():
    out = compute_technical(_frame([100.0, 110.0, 99.0, 99.0]))
    np.testing.assert_allclose(
        out["ret_1"].to_numpy(dtype=float),
        [np.nan, 0.10, -0.10, 0.0],
        equal_nan=True,
        rtol=1e-9,
    )


def test_sma_deviation_constant_series_is_zero():
    # 一定値なら close == SMA なので乖離は 0（ウォームアップ後）。
    out = compute_technical(_frame([100.0] * 25))
    np.testing.assert_allclose(out["sma5_dev"].to_numpy()[4:], 0.0, atol=1e-12)
    np.testing.assert_allclose(out["sma20_dev"].to_numpy()[19:], 0.0, atol=1e-12)
    # ウォームアップ区間は NaN。
    assert np.isnan(out["sma20_dev"].to_numpy()[:19]).all()


def test_rsi_bounds_and_all_up():
    # 単調増加 → 損失が無いので RSI は 100（ウォームアップ後）。
    out = compute_technical(_frame([100.0 + i for i in range(40)]))
    rsi = out["rsi14"].to_numpy(dtype=float)
    np.testing.assert_allclose(rsi[14:], 100.0, atol=1e-9)
    assert np.isnan(rsi[:14]).all()
    # 値域 [0, 100]（NaN を除く）。
    valid = rsi[~np.isnan(rsi)]
    assert ((valid >= 0.0) & (valid <= 100.0)).all()


def test_rsi_all_down_is_zero():
    out = compute_technical(_frame([200.0 - i for i in range(40)]))
    rsi = out["rsi14"].to_numpy(dtype=float)
    np.testing.assert_allclose(rsi[14:], 0.0, atol=1e-9)


def test_macd_zero_on_constant_series():
    # 一定値なら MACD もシグナルも 0 に収束する。
    out = compute_technical(_frame([100.0] * 60))
    np.testing.assert_allclose(out["macd"].to_numpy()[26:], 0.0, atol=1e-9)
    np.testing.assert_allclose(out["macd_hist"].to_numpy()[40:], 0.0, atol=1e-9)


def test_no_lookahead_truncation_invariance():
    """系列を未来側で切っても、過去の特徴量が完全一致すること（リーク無しの根拠）。"""
    rng = np.random.default_rng(42)
    closes = (100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, size=80))).tolist()
    volumes = (rng.integers(500, 5000, size=80).astype(float)).tolist()

    full = compute_technical(_frame(closes, volumes=volumes))
    cut = compute_technical(_frame(closes[:50], volumes=volumes[:50]))

    pd.testing.assert_frame_equal(
        full.iloc[:50].reset_index(drop=True),
        cut.reset_index(drop=True),
    )


def test_per_code_independence():
    """複数銘柄を一括投入しても、特徴量が銘柄内で閉じて計算されること。"""
    up = _frame([100.0 + i for i in range(40)], code="UP")
    down = _frame([200.0 - i for i in range(40)], code="DOWN")
    out = compute_technical(pd.concat([up, down], ignore_index=True))

    up_rsi = out.loc[out["code"] == "UP", "rsi14"].to_numpy(dtype=float)
    down_rsi = out.loc[out["code"] == "DOWN", "rsi14"].to_numpy(dtype=float)
    np.testing.assert_allclose(up_rsi[14:], 100.0, atol=1e-9)
    np.testing.assert_allclose(down_rsi[14:], 0.0, atol=1e-9)
