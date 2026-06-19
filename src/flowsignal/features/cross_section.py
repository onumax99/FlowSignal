"""クロスセクション（相対）予測のための変換（prediction-design §4②）。

絶対方向ではなく「**同日内でどの銘柄が相対的に強いか**」を予測対象にする:
- 目的変数 = 翌日リターンの **日次デミーン**（市場共通要因＝日次でほぼ予測不能を除いた市場中立リターン）。
- 特徴量 = **日次クロスセクションで z-score**。マーケット・カレンダー特徴量は同日内で全銘柄一定（std=0）
  なので 0 になり、銘柄固有のテクニカルだけが相対予測に効くようになる。

リーク防止:
- デミーン / z-score はいずれも **同日 t の断面のみ**で計算（未来を見ない）。
  未来を見るのは目的変数 ret_fwd（= t+1）だけで、これは従来どおり。
"""

from __future__ import annotations

import pandas as pd


def add_relative_target(
    df: pd.DataFrame, *, ret_col: str = "ret_fwd", out_col: str = "rel_fwd", min_names: int = 2
) -> pd.DataFrame:
    """``ret_col`` を日次デミーンした市場中立リターンを ``out_col`` として付与する。

    断面の銘柄数が ``min_names`` 未満の日は NaN（相対化できないため）。
    """
    grp = df.groupby("date")[ret_col]
    rel = df[ret_col] - grp.transform("mean")
    rel = rel.where(grp.transform("count") >= min_names)
    return df.assign(**{out_col: rel})


def cross_sectional_zscore(
    df: pd.DataFrame, feature_cols: list[str], *, date_col: str = "date"
) -> pd.DataFrame:
    """各日付の断面で ``feature_cols`` を z-score（mean 0 / std 1）した DataFrame を返す。

    同日内で一定の列（std=0、例: マーケット/カレンダー）は 0 にする。ウォームアップ NaN は保持。
    返り値カラム: date, code, <feature_cols...>。
    """
    out = df[[date_col, "code"]].copy()
    grp = df.groupby(date_col)
    for col in feature_cols:
        mean = grp[col].transform("mean")
        std = grp[col].transform("std")
        z = (df[col] - mean) / std
        out[col] = z.where(std > 0, 0.0)  # std=0/NaN（断面一定や単一銘柄）→ 0
    return out.reset_index(drop=True)
