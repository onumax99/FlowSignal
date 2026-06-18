"""HOLD（確信度で棄権）と確率較正(Platt)の単体テスト。"""

from __future__ import annotations

import numpy as np
import pytest

from flowsignal.eval.calibration import PlattCalibrator
from flowsignal.eval.hold import (
    confidence,
    coverage_curve,
    covered_metrics,
    select_threshold_for_coverage,
)
from flowsignal.features.labels import LABEL_CLASSES


# --- PlattCalibrator ---------------------------------------------------------


def _proba_and_labels(n=200, seed=0):
    rng = np.random.default_rng(seed)
    proba = rng.dirichlet([1.0, 1.0, 1.0], size=n)  # 各行 sum=1
    y = np.array(LABEL_CLASSES)[proba.argmax(axis=1)]
    return proba, y


def test_calibrator_outputs_are_normalized_probabilities():
    proba, y = _proba_and_labels()
    out = PlattCalibrator(LABEL_CLASSES).fit(proba, y).transform(proba)
    assert out.shape == proba.shape
    assert np.allclose(out.sum(axis=1), 1.0)
    assert (out >= 0).all()


def test_calibrator_handles_class_absent_in_cal():
    proba, y = _proba_and_labels()
    y_no_down = np.where(y == "DOWN", "FLAT", y)  # DOWN が cal に無いケース
    out = PlattCalibrator(LABEL_CLASSES).fit(proba, y_no_down).transform(proba)
    assert np.allclose(out.sum(axis=1), 1.0)
    # DOWN は定数 0（出現割合 0）→ 正規化後も 0
    assert np.allclose(out[:, LABEL_CLASSES.index("DOWN")], 0.0)


# --- confidence / threshold --------------------------------------------------


def test_confidence_is_row_max():
    proba = np.array([[0.2, 0.5, 0.3], [0.7, 0.1, 0.2]])
    assert confidence(proba).tolist() == [0.5, 0.7]


def test_threshold_full_coverage_is_zero():
    conf = np.linspace(0.4, 1.0, 50)
    assert select_threshold_for_coverage(conf, 1.0) == 0.0


def test_threshold_hits_target_coverage():
    conf = np.linspace(0.0, 1.0, 100)
    t = select_threshold_for_coverage(conf, 0.5)
    assert (conf >= t).mean() == pytest.approx(0.5, abs=0.02)


def test_threshold_monotonic_in_coverage():
    conf = np.linspace(0.0, 1.0, 100)
    # 被覆率を下げる（厳しくする）ほど閾値は上がる
    assert select_threshold_for_coverage(conf, 0.3) >= select_threshold_for_coverage(
        conf, 0.7
    )


# --- covered metrics / curve -------------------------------------------------


def test_covered_metrics_full_and_perfect():
    y = np.array(LABEL_CLASSES * 10)  # 30 行
    conf = np.linspace(0.4, 1.0, 30)
    m = covered_metrics(y, y, conf, 0.0)  # 全件・完全予測
    assert m["coverage"] == 1.0 and m["n"] == 30
    assert m["macro_f1"] == 1.0
    assert sum(m["pred_mix"].values()) == pytest.approx(1.0)


def test_covered_metrics_abstains_above_threshold():
    y = np.array(LABEL_CLASSES * 10)
    conf = np.linspace(0.4, 1.0, 30)
    assert covered_metrics(y, y, conf, 0.9)["coverage"] < 1.0


def test_covered_metrics_empty_is_nan():
    y = np.array(LABEL_CLASSES * 10)
    conf = np.linspace(0.4, 1.0, 30)
    m = covered_metrics(y, y, conf, 2.0)  # 誰も閾値を超えない
    assert m["n"] == 0 and np.isnan(m["macro_f1"])


def test_coverage_curve_is_monotonic_in_coverage():
    y = np.array(LABEL_CLASSES * 10)
    conf = np.linspace(0.4, 1.0, 30)
    covs = [c["coverage"] for c in coverage_curve(y, y, conf, [0.0, 0.5, 0.9, 1.1])]
    assert covs == sorted(covs, reverse=True)
