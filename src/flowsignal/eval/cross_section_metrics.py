"""クロスセクション（相対）予測の評価指標。

3 クラス accuracy/macro-F1 ではなく、相対予測に合う指標で測る（prediction-design §4②）:
- **日次 rank IC**: 各日について「予測スコア」と「実現の相対リターン」の順位相関（Spearman）。
  mean IC / ICIR（mean/std）/ t 値（ICIR×√日数）/ 正の日割合 を見る。
- **ロングショート・スプレッド**: 各日スコア上位 q を買い・下位 q を売った実現リターン差。
  日次平均と簡易 Sharpe（√252 で年率化）。

いずれも **日（断面）を 1 単位**にする＝同一日の銘柄間相関を持ち込まない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def daily_rank_ic(scores, targets, dates, *, min_names: int = 5) -> dict:
    """日次 rank IC（Spearman）の集計。ics に各日の IC 配列も返す。"""
    df = pd.DataFrame({"score": np.asarray(scores), "target": np.asarray(targets),
                       "date": np.asarray(dates)}).dropna()
    ics = []
    for _, g in df.groupby("date"):
        if len(g) < min_names:
            continue
        ic, _p = spearmanr(g["score"], g["target"])
        if not np.isnan(ic):
            ics.append(float(ic))
    ics = np.asarray(ics, dtype="float64")
    n = len(ics)
    mean = float(ics.mean()) if n else float("nan")
    std = float(ics.std(ddof=1)) if n > 1 else float("nan")
    icir = mean / std if (std and std > 0 and not np.isnan(std)) else float("nan")
    t_stat = icir * np.sqrt(n) if (n and not np.isnan(icir)) else float("nan")
    pos = float((ics > 0).mean()) if n else float("nan")
    return {"mean_ic": mean, "std_ic": std, "icir": icir, "t_stat": t_stat,
            "pos_frac": pos, "n_days": n, "ics": ics}


def long_short_returns(scores, targets, dates, *, quantile: float = 0.2,
                       min_names: int = 5) -> dict:
    """日次ロングショート（スコア上位 q 買い・下位 q 売り）の実現リターン差。"""
    df = pd.DataFrame({"score": np.asarray(scores), "target": np.asarray(targets),
                       "date": np.asarray(dates)}).dropna()
    spreads, longs, shorts = [], [], []
    for _, g in df.groupby("date"):
        if len(g) < min_names:
            continue
        k = max(1, int(round(len(g) * quantile)))
        gg = g.sort_values("score")
        lo = float(gg.head(k)["target"].mean())   # スコア下位（売り）
        hi = float(gg.tail(k)["target"].mean())   # スコア上位（買い）
        spreads.append(hi - lo)
        longs.append(hi)
        shorts.append(lo)
    spreads = np.asarray(spreads, dtype="float64")
    n = len(spreads)
    mean = float(spreads.mean()) if n else float("nan")
    std = float(spreads.std(ddof=1)) if n > 1 else float("nan")
    sharpe = mean / std * np.sqrt(252) if (std and std > 0) else float("nan")
    return {"mean_spread": mean, "std_spread": std, "sharpe": sharpe, "n_days": n,
            "mean_long": float(np.mean(longs)) if longs else float("nan"),
            "mean_short": float(np.mean(shorts)) if shorts else float("nan")}


def bootstrap_mean_ci(values, *, n_boot: int = 2000, seed: int = 42,
                      lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    """値（例: 各日の IC）の平均の bootstrap 信頼区間。日を 1 単位に重複抽出。"""
    v = np.asarray(values, dtype="float64")
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return (float(np.percentile(means, lo)), float(np.percentile(means, hi)))
