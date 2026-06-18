"""HOLD（確信度で棄権）の評価ロジック。

全日当てにいかず**確信度が閾値以上の行だけ予測**し、残りは棄権（予測しない）。
評価は「カバレッジ（予測割合）× その精度」のトレードオフで見る。

⚠️ 落とし穴（prediction-design §4①）:
- 棄権するとモデルが自信を持つ多数派(FLAT)帯に偏りがちで、**accuracy は自動で上がる**。
  → 本命は **coverage × macro-F1**。さらに **covered set の予測クラス構成**も併記し FLAT 偏りを露出させる。
- 閾値を **test 上で best に選ぶと楽観バイアス**。閾値は validation の確信度で被覆率目標に合わせて決め、
  test に適用する（select_threshold_for_coverage を validation 確信度に対して使う）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flowsignal.eval.metrics import classification_metrics
from flowsignal.features.labels import LABEL_CLASSES


def confidence(proba) -> np.ndarray:
    """各行の確信度＝予測確率の最大値。"""
    return np.asarray(proba, dtype="float64").max(axis=1)


def select_threshold_for_coverage(conf, target_coverage: float) -> float:
    """確信度 ``conf`` で被覆率 ``target_coverage`` を満たす閾値を返す。

    **validation の確信度**に対して使い、得た閾値を test に適用する（リーク防止）。
    target_coverage>=1 は全件予測（閾値 0）。
    """
    if target_coverage >= 1.0:
        return 0.0
    if target_coverage <= 0.0:
        return float("inf")
    c = np.sort(np.asarray(conf, dtype="float64"))[::-1]  # 降順
    k = int(np.ceil(target_coverage * len(c)))
    k = min(max(k, 1), len(c))
    return float(c[k - 1])  # 上位 k 番目の確信度＝この閾値以上が概ね target_coverage


def covered_metrics(y_true, y_pred, conf, threshold: float, classes=None) -> dict:
    """確信度 >= threshold の行だけで accuracy / macro-F1 と予測クラス構成を出す。"""
    classes = classes or LABEL_CLASSES
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    conf = np.asarray(conf, dtype="float64")
    mask = conf >= threshold
    n = int(mask.sum())
    coverage = float(mask.mean()) if len(conf) else 0.0
    if n == 0:
        return {
            "threshold": float(threshold), "coverage": coverage, "n": 0,
            "accuracy": float("nan"), "macro_f1": float("nan"),
            "pred_mix": {c: 0.0 for c in classes},
        }
    m = classification_metrics(y_true[mask], y_pred[mask], classes)
    mix = pd.Series(y_pred[mask]).value_counts(normalize=True)
    return {
        "threshold": float(threshold), "coverage": coverage, "n": n,
        "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
        "pred_mix": {c: float(mix.get(c, 0.0)) for c in classes},
    }


def coverage_curve(y_true, y_pred, conf, thresholds, classes=None) -> list[dict]:
    """各閾値での covered_metrics を並べた記述的トレードオフ曲線。"""
    return [covered_metrics(y_true, y_pred, conf, t, classes) for t in thresholds]
