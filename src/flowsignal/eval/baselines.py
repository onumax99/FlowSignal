"""3 つの必須ベースライン（モデルが有意に上回るべき対象）。

- always_up      : 常に "UP"（上昇相場で accuracy が見かけ上高くなりがち）。
- random         : 3 クラス一様ランダム（seed 固定で再現可能）。
- prev_direction : 前日同方向（persistence）。row t の予測 = その銘柄の
                   1 つ前の実現ラベル（= t-1→t の方向, t 時点で既知）。リーク無し。

いずれも df の index に整合した予測 Series を返す。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flowsignal.features.labels import LABEL_CLASSES
from flowsignal.models.baseline import RANDOM_SEED


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
