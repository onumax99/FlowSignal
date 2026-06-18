"""確率較正（Platt スケーリング, 多クラス 1-vs-rest）。

lightgbm の予測確率はそのままだと確信度として当てにならないため、**検証区間で較正**してから
HOLD（確信度で棄権）に使う。小標本では isotonic は過学習しやすいので **Platt（シグモイド）を既定**。

リーク防止（最重要）:
- 較正器は **train 末尾の validation 区間で fit** し、test では fit しない（時系列順守）。
- → fold ごとに「fit 区間でモデル学習 → cal 区間で較正器 fit → test で適用」。

各クラス c について 1-vs-rest のロジスティック回帰 sigmoid(a_c·p_c + b_c) を当てはめ、
全クラスで正規化して確率に戻す。あるクラスが cal 区間に1つも無い場合は定数（出現割合）で代替。
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """多クラス Platt スケーリング（クラスごと 1-vs-rest シグモイド＋正規化）。

    使い方:
        cal = PlattCalibrator(model.classes_).fit(model.predict_proba(X_cal), y_cal)
        proba = cal.transform(model.predict_proba(X_test))
    """

    def __init__(self, classes):
        self.classes = list(classes)
        self._fitted: dict[str, tuple] = {}

    def fit(self, proba, y) -> "PlattCalibrator":
        proba = np.asarray(proba, dtype="float64")
        y = np.asarray(y)
        for i, c in enumerate(self.classes):
            target = (y == c).astype(int)
            if len(np.unique(target)) < 2:
                # cal にこのクラスが無い（or 全部）→ 定数（出現割合）で代替
                self._fitted[c] = ("const", float(target.mean()))
            else:
                lr = LogisticRegression()
                lr.fit(proba[:, [i]], target)
                self._fitted[c] = ("lr", lr)
        return self

    def transform(self, proba) -> np.ndarray:
        proba = np.asarray(proba, dtype="float64")
        cols = []
        for i, c in enumerate(self.classes):
            kind, m = self._fitted[c]
            if kind == "const":
                cols.append(np.full(len(proba), m))
            else:
                cols.append(m.predict_proba(proba[:, [i]])[:, 1])
        cal = np.column_stack(cols)
        s = cal.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0  # 全クラス 0 の行は均すため割らない
        return cal / s
