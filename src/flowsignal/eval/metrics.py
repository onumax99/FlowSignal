"""分類指標とベースライン比較・有意性チェック。

提供する指標（要件 §9 / STATUS §6.2）:
- accuracy, macro-F1（FLAT 偏重に騙されないため accuracy 単独で判断しない）
- per-class precision / recall / f1 / support, 混同行列
- **McNemar 検定**: モデル vs ベースライン（prev-direction 等）の accuracy 有意差。
  注: McNemar は accuracy（正誤の 2×2）に対する検定であり macro-F1 は対象外。
  fold 間の mean±std は fold が相関するため厳密な検定ではない（PoC のヒューリスティック）。
- **macro-F1 の日付ブロック bootstrap**: 本命指標 macro-F1 の CI と差の有意性
  （McNemar が使えない macro-F1 用。再標本化は日付ブロック単位 = block_bootstrap_macro_f1）。
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


def _encode(y, code: dict[str, int]) -> np.ndarray:
    """ラベル配列を整数コードへ（高速 bootstrap 用）。"""
    arr = np.asarray(y)
    out = np.empty(len(arr), dtype=np.int64)
    for label, i in code.items():
        out[arr == label] = i
    return out


def _fast_macro_f1(yt: np.ndarray, yp: np.ndarray, n_classes: int) -> float:
    """整数コード済み配列から macro-F1 を計算（sklearn より軽量・bootstrap ループ用）。

    sklearn の ``f1_score(..., average="macro", labels=全クラス, zero_division=0)`` と一致。
    """
    cm = np.bincount(yt * n_classes + yp, minlength=n_classes * n_classes)
    cm = cm.reshape(n_classes, n_classes)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(denom > 0, 2 * tp / denom, 0.0)
    return float(f1.mean())


def block_bootstrap_macro_f1(
    y_true,
    pred_dict: dict,
    blocks,
    *,
    reference: str | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    labels: list[str] | None = None,
) -> dict:
    """日付などのブロック単位で resample した macro-F1 の bootstrap CI。

    McNemar は accuracy 専用なので、本命指標 macro-F1 の有意性はこちらで見る。
    再標本化の単位は **blocks（= 日付を渡す）**。同一日の銘柄間相関を保ったまま日を
    重複ありで抽出するため、行単位 bootstrap のように CI が楽観的にならない。
    **5 fold を resample するのではない**点に注意（prediction-design §4 / m2-evaluation 限界#3）。

    Args:
        pred_dict: ``{名前: 予測ラベル配列}``。``reference`` を含めると各系列との差も出す。
        blocks   : 各行のブロック ID（通常は date）。長さは y_true と一致。
        reference: 差分 CI の基準（通常 "model"）。None なら各系列の CI のみ。

    Returns:
        ``{"n_boot", "point": {name: macroF1}, "ci": {name: (lo, hi)},
          "diff_vs_reference": {"reference", "vs": {name: {diff, ci_low, ci_high, p_one_sided}}}}``
        - ``ci`` は各系列 macro-F1 の 95%（2.5/97.5 パーセンタイル）区間。
        - ``diff`` = reference − name の点推定。``ci_low/high`` がともに > 0 なら有意に上。
        - ``p_one_sided`` = resample で diff ≤ 0 となった割合（0 に近いほど reference 優位）。
    """
    labels = labels or LABEL_CLASSES
    n_classes = len(labels)
    code = {c: i for i, c in enumerate(labels)}
    yt = _encode(y_true, code)
    preds = {k: _encode(v, code) for k, v in pred_dict.items()}

    blocks = np.asarray(blocks)
    if len(blocks) != len(yt):
        raise ValueError("blocks の長さが y_true と一致しません。")
    uniq, inv = np.unique(blocks, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
    block_rows = [order[starts[i] : starts[i + 1]] for i in range(len(uniq))]
    n_blocks = len(uniq)

    rng = np.random.default_rng(seed)
    names = list(pred_dict)
    draws = {k: np.empty(n_boot, dtype=float) for k in names}
    for b in range(n_boot):
        sampled = rng.integers(0, n_blocks, size=n_blocks)
        idx = np.concatenate([block_rows[s] for s in sampled])
        yt_b = yt[idx]
        for k in names:
            draws[k][b] = _fast_macro_f1(yt_b, preds[k][idx], n_classes)

    point = {k: _fast_macro_f1(yt, preds[k], n_classes) for k in names}
    ci = {
        k: (float(np.percentile(draws[k], 2.5)), float(np.percentile(draws[k], 97.5)))
        for k in names
    }
    result = {"n_boot": int(n_boot), "point": point, "ci": ci}

    if reference is not None:
        if reference not in pred_dict:
            raise KeyError(f"reference '{reference}' が pred_dict にありません。")
        vs = {}
        for k in names:
            if k == reference:
                continue
            d = draws[reference] - draws[k]
            lo, hi = np.percentile(d, [2.5, 97.5])
            vs[k] = {
                "diff": float(point[reference] - point[k]),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "p_one_sided": float(np.mean(d <= 0.0)),
            }
        result["diff_vs_reference"] = {"reference": reference, "vs": vs}
    return result
