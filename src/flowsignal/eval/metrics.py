"""分類指標とベースライン比較・有意性チェック。

提供する指標（要件 §9 / STATUS §6.2）:
- accuracy, macro-F1（FLAT 偏重に騙されないため accuracy 単独で判断しない）
- per-class precision / recall / f1 / support, 混同行列
- **McNemar 検定**: モデル vs ベースライン（prev-direction 等）の accuracy 有意差。
  注: McNemar は accuracy（正誤の 2×2）に対する検定であり macro-F1 は対象外。
  fold 間の mean±std は fold が相関するため厳密な検定ではない（PoC のヒューリスティック）。
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from flowsignal.features.labels import LABEL_CLASSES


def classification_metrics(y_true, y_pred, labels: list[str] | None = None) -> dict:
    """accuracy・macro-F1・per-class・混同行列をまとめた dict を返す。"""
    labels = labels or LABEL_CLASSES
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "per_class": {
            labels[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(labels))
        },
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": list(labels),
    }


def mcnemar_test(y_true, pred_model, pred_baseline) -> dict:
    """モデル vs ベースラインの McNemar 検定（連続性補正あり, df=1）。

    Returns:
        n10（モデル正・ベース誤）, n01（モデル誤・ベース正）, statistic, pvalue。
    """
    y_true = np.asarray(y_true)
    model_correct = np.asarray(pred_model) == y_true
    base_correct = np.asarray(pred_baseline) == y_true
    n10 = int(np.sum(model_correct & ~base_correct))
    n01 = int(np.sum(~model_correct & base_correct))
    discordant = n10 + n01
    if discordant == 0:
        return {"n10": n10, "n01": n01, "statistic": 0.0, "pvalue": 1.0}
    statistic = (abs(n10 - n01) - 1) ** 2 / discordant  # 連続性補正
    pvalue = float(chi2.sf(statistic, df=1))
    return {
        "n10": n10,
        "n01": n01,
        "statistic": float(statistic),
        "pvalue": pvalue,
    }
