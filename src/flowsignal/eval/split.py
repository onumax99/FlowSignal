"""時系列分割（walk-forward）。**分割は必ず「日付境界」で行う。**

リーク防止の要点（prediction-design §2.1 / STATUS §6.2）:
- 全銘柄プールを行単位で分割すると、**同一日付の別銘柄が train/test をまたいで
  リーク**する。よって時刻 t の全銘柄は必ず同じ fold に入れる＝**カレンダー日付基準**で割る。
- walk-forward（expanding window）: train は過去側に拡大、test は連続ブロックを前進。
  各 fold で max(train 日付) < min(test 日付) を保証する。

使い方:
    for train_mask, test_mask in split_masks(df, n_splits=5):
        Xtr, Xte = df[train_mask], df[test_mask]
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import pandas as pd


def walk_forward_date_splits(
    dates: Iterable, n_splits: int = 5
) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """ユニーク日付を expanding walk-forward に分割し、(train_dates, test_dates) を返す。"""
    if n_splits < 1:
        raise ValueError("n_splits は 1 以上で指定してください。")
    u = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Index(dates)).unique()))
    n = len(u)
    if n < n_splits + 1:
        raise ValueError(
            f"日付数({n})が分割数({n_splits})に対して少なすぎます（{n_splits + 1} 日以上必要）。"
        )
    fold = n // (n_splits + 1)
    for i in range(1, n_splits + 1):
        train_end = fold * i
        test_end = n if i == n_splits else fold * (i + 1)
        yield u[:train_end], u[train_end:test_end]


def split_masks(
    df: pd.DataFrame, n_splits: int = 5, date_col: str = "date"
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """df の各行に対する (train_mask, test_mask) の真偽配列を fold ごとに返す。"""
    d = pd.to_datetime(df[date_col])
    for train_dates, test_dates in walk_forward_date_splits(df[date_col], n_splits):
        train_mask = d.isin(set(train_dates)).to_numpy()
        test_mask = d.isin(set(test_dates)).to_numpy()
        yield train_mask, test_mask
