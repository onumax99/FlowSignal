"""必須ベースライン（モデルが有意に上回るべき対象）。

- always_majority: 最頻クラス（不均衡 3 クラスでは最強の trivial baseline）。
                   多数派は **train 区間から決める**（test を見て多数派を選ぶと軽微なリーク）。
                   → fold ごとに majority_class(y_train) を使う（train_baseline 側で処理）。
- always_up      : 常に "UP"（上昇相場で accuracy が見かけ上高くなりがち）。
- random         : 3 クラス一様ランダム（seed 固定で再現可能）。
- prev_direction : 前日同方向（persistence）。row t の予測 = その銘柄の
                   1 つ前の実現ラベル（= t-1→t の方向, t 時点で既知）。リーク無し。

accuracy では always_majority が最強・macro-F1 では random/prev_direction が最強、と
**指標で強弱が入れ替わる**。両指標を併記して対称に比較する（m2-evaluation 限界#1/#2）。
baseline_predictions は always_up / random / prev_direction（train 不要）を返す。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flowsignal.features.labels import LABEL_CLASSES
from flowsignal.models.baseline import RANDOM_SEED


def majority_class(labels, *, classes: list[str] = LABEL_CLASSES) -> str:
    """最頻クラスを返す（多数派ベースライン用）。

    タイは ``classes`` の並び順で先勝ち（決定的）。**train 区間のラベルのみ**を渡すこと
    （test を見て多数派を決めるとリークになるため）。
    """
    counts = pd.Series(list(labels)).dropna().value_counts()
    return max(classes, key=lambda c: (int(counts.get(c, 0)), -classes.index(c)))


def baseline_predictions(
    df: pd.DataFrame,
    *,
    seed: int = RANDOM_SEED,
    label_col: str = "label",
    code_col: str = "code",
) -> pd.DataFrame:
    """always_up / random / prev_direction の予測列を持つ DataFrame を返す。

    prev_direction は銘柄ごとに 1 つ前の実現ラベルを使う（先頭行は欠損→FLAT で穴埋め）。
    """
    out = pd.DataFrame(index=df.index)
    out["always_up"] = "UP"

    rng = np.random.default_rng(seed)
    out["random"] = rng.choice(LABEL_CLASSES, size=len(df))

    prev = df.groupby(code_col)[label_col].shift(1)
    out["prev_direction"] = prev.fillna("FLAT")
    return out
