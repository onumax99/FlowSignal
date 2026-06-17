"""labels.py の単体テスト。

検証ポイント:
- 固定閾値での 3 クラス判定が定義どおり。
- 各銘柄の最終日（t+1 が無い）はラベル NaN。
- vol モードのウォームアップ（σ 未定義区間）はラベル NaN。
- **閾値 σ_t は後方参照のみ**（未来側を切っても閾値が変わらない＝リーク無し）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flowsignal.features.labels import (
    LABEL_CLASSES,
    class_distribution,
    compute_labels,
)


def _frame(closes, code="AAA", start="2024-01-01") -> pd.DataFrame:
    n = len(closes)
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {"date": dates, "code": code, "close": [float(c) for c in closes]}
    )


def test_output_schema():
    out = compute_labels(_frame([100.0] * 30))
    assert list(out.columns) == ["date", "code", "ret_fwd", "threshold", "label"]


def test_required_columns_validated():
    with pytest.raises(ValueError):
        compute_labels(pd.DataFrame({"date": [], "code": []}))


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        compute_labels(_frame([100.0, 101.0, 102.0]), mode="bad")


def test_fixed_mode_known_values():
    # ret_fwd = [+0.015, 0.0, -0.0148, 0.0, NaN], 閾値 ±0.01。
    out = compute_labels(
        _frame([100.0, 101.5, 101.5, 100.0, 100.0]),
        mode="fixed",
        fixed_threshold=0.01,
    )
    labels = out["label"].tolist()
    assert labels[:4] == ["UP", "FLAT", "DOWN", "FLAT"]
    assert pd.isna(labels[4])  # 最終日は t+1 が無く NaN


def test_last_row_per_code_is_nan():
    out = compute_labels(_frame([100.0, 110.0, 120.0]), mode="fixed")
    assert pd.isna(out["label"].iloc[-1])
    assert pd.isna(out["ret_fwd"].iloc[-1])


def test_vol_mode_warmup_and_classes():
    rng = np.random.default_rng(0)
    closes = (100.0 * np.cumprod(1.0 + rng.normal(0, 0.02, 30))).tolist()
    out = compute_labels(_frame(closes), mode="vol", k=0.5, vol_window=20)
    labels = out["label"]
    # σ ウォームアップ（先頭 20 行）はラベル NaN。
    assert labels.iloc[:20].isna().all()
    # 有効区間（row 20..28）は 3 クラスのいずれか、最終行(29)は NaN。
    valid = labels.iloc[20:29]
    assert valid.notna().all()
    assert set(valid.unique()).issubset(set(LABEL_CLASSES))
    assert pd.isna(labels.iloc[29])


def test_threshold_no_lookahead():
    """閾値 σ_t は未来を見ない: 系列を切っても過去の threshold が一致する。"""
    rng = np.random.default_rng(3)
    closes = (100.0 * np.cumprod(1.0 + rng.normal(0, 0.02, 60))).tolist()

    full = compute_labels(_frame(closes), mode="vol", vol_window=20)
    cut = compute_labels(_frame(closes[:40]), mode="vol", vol_window=20)

    # threshold は ≤ t のみ依存 → 先頭 40 行が完全一致。
    np.testing.assert_allclose(
        full["threshold"].to_numpy(dtype=float)[:40],
        cut["threshold"].to_numpy(dtype=float),
        equal_nan=True,
    )
    # ラベルは t+1 を要するため、t+1 が切られていない 39 行目までで一致
    # （NaN セーフ比較のため欠損を文字列で穴埋め）。
    assert (
        full["label"].iloc[:39].fillna("NA").tolist()
        == cut["label"].iloc[:39].fillna("NA").tolist()
    )


def test_per_code_independence():
    up = _frame([100.0 + 3 * i for i in range(25)], code="UP")
    down = _frame([200.0 - 3 * i for i in range(25)], code="DOWN")
    out = compute_labels(
        pd.concat([up, down], ignore_index=True), mode="fixed", fixed_threshold=0.005
    )
    up_labels = out.loc[out["code"] == "UP", "label"].dropna()
    down_labels = out.loc[out["code"] == "DOWN", "label"].dropna()
    assert (up_labels == "UP").all()
    assert (down_labels == "DOWN").all()


def test_class_distribution_helper():
    s = pd.Series(["UP", "UP", "FLAT", "DOWN", np.nan])
    dist = class_distribution(s)
    assert dist["UP"] == 2 and dist["FLAT"] == 1 and dist["DOWN"] == 1
    assert list(dist.index) == LABEL_CLASSES
