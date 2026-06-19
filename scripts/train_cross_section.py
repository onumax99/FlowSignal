"""M2.5 クロスSECTION（相対）予測の学習〜評価の通し実行。

絶対方向ではなく「同日内でどの銘柄が相対的に強いか」を予測する（prediction-design §4②）:
- 目的変数 = 翌日リターンの日次デミーン（市場中立）。
- 特徴量 = 日次クロスセクションで z-score（マーケット/カレンダーは断面一定→0、銘柄固有テクニカルが効く）。
- モデル = LightGBM 回帰。日付境界 walk-forward。
- 評価 = 日次 rank IC（mean/ICIR/t 値/正の日割合・日ブートストラップ CI）＋ ロングショート・スプレッド。
  ベースライン = ランダムスコア（IC≈0）と 5 日モメンタム（cs z-score 済み ret_5）。

使い方:
    python scripts/train_cross_section.py
    python scripts/train_cross_section.py --n-splits 5 --quantile 0.2 --n-boot 2000
"""

from __future__ import annotations

import argparse

import numpy as np

from flowsignal.eval.cross_section_metrics import (
    bootstrap_mean_ci,
    daily_rank_ic,
    long_short_returns,
)
from flowsignal.eval.split import split_masks
from flowsignal.features.build import FEATURE_COLUMNS, load_and_build
from flowsignal.features.cross_section import add_relative_target, cross_sectional_zscore
from flowsignal.models.baseline import RANDOM_SEED, make_regressor


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FlowSignal M2.5 クロスセクション相対予測")
    p.add_argument("--label-mode", choices=["vol", "fixed"], default="vol")
    p.add_argument("--k", type=float, default=0.5)
    p.add_argument("--vol-window", type=int, default=20)
    p.add_argument("--model", choices=["lightgbm", "hist"], default="lightgbm")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--quantile", type=float, default=0.2, help="ロングショートの上位/下位割合")
    p.add_argument("--n-boot", type=int, default=2000, help="mean IC の日ブートストラップ反復")
    return p.parse_args()


def run(args: argparse.Namespace) -> dict:
    base = load_and_build(label_mode=args.label_mode, k=args.k, vol_window=args.vol_window)
    base = add_relative_target(base)
    base = base[base["rel_fwd"].notna()].reset_index(drop=True)

    Xz = cross_sectional_zscore(base, FEATURE_COLUMNS)
    X = Xz[FEATURE_COLUMNS]
    y = base["rel_fwd"]
    dates = base["date"]

    pooled = {"score": [], "target": [], "date": [], "mom": []}
    for tr, te in split_masks(base, n_splits=args.n_splits):
        model = make_regressor(args.model, seed=args.seed)
        model.fit(X[tr], y[tr])
        pooled["score"].append(np.asarray(model.predict(X[te])))
        pooled["target"].append(y[te].to_numpy())
        pooled["date"].append(dates[te].to_numpy())
        pooled["mom"].append(X["ret_5"][te].to_numpy())  # cs z-score 済みの 5 日モメンタム

    return {k: np.concatenate(v) for k, v in pooled.items()}


def _fmt_ic(ic: dict, ci: tuple[float, float] | None = None) -> str:
    s = (f"mean IC={ic['mean_ic']:+.4f}  ICIR={ic['icir']:+.3f}  t={ic['t_stat']:+.2f}  "
         f"正の日={ic['pos_frac']*100:.0f}%  (n_days={ic['n_days']})")
    if ci is not None:
        s += f"  95%CI[{ci[0]:+.4f}, {ci[1]:+.4f}]"
    return s


def main() -> None:
    args = _parse_args()
    res = run(args)
    score, target, date, mom = res["score"], res["target"], res["date"], res["mom"]
    print(f"[cross-section] n={len(score)} 行 / walk-forward {args.n_splits} folds / "
          f"目的=翌日リターンの日次デミーン / 特徴量=日次 z-score\n")

    ic = daily_rank_ic(score, target, date)
    ci = bootstrap_mean_ci(ic["ics"], n_boot=args.n_boot, seed=args.seed)
    print("[rank IC] モデル: " + _fmt_ic(ic, ci))

    rng = np.random.default_rng(args.seed)
    ic_rand = daily_rank_ic(rng.normal(size=len(score)), target, date)
    ic_mom = daily_rank_ic(mom, target, date)
    print("[rank IC] ランダム: " + _fmt_ic(ic_rand))
    print("[rank IC] モメンタム(ret_5): " + _fmt_ic(ic_mom))

    sig = "有意（CI が 0 を外す）" if (ci[0] > 0 or ci[1] < 0) else "有意差なし（CI が 0 を跨ぐ）"
    print(f"  → mean IC の 95%CI は {sig}")

    ls = long_short_returns(score, target, date, quantile=args.quantile)
    print(f"\n[ロングショート] 上位/下位 {args.quantile:.0%} / 日次・市場中立リターン")
    print(f"    日次スプレッド平均={ls['mean_spread']*1e4:+.1f}bp  "
          f"Sharpe(√252)={ls['sharpe']:+.2f}  (long {ls['mean_long']*1e4:+.1f}bp / "
          f"short {ls['mean_short']*1e4:+.1f}bp, n_days={ls['n_days']})")

    print("\n  読み方: mean IC>0 かつ CI が 0 を外せば相対予測に edge。日次 IC 0.01〜0.03 でも"
          " 有意なら実用域（クロスセクションは小さな IC を多銘柄×多日で積む）。")


if __name__ == "__main__":
    main()
