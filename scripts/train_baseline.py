"""M2 ベースライン学習〜評価の通し実行。

テクニカルのみで翌営業日 3 クラス方向を予測し、**3 つのベースライン
（always-up / random / prev-direction）と比較**して有意に上回るかを検証する。

使い方:
    python scripts/train_baseline.py                    # 既定: ボラ連動ラベル, lightgbm, 5-fold
    python scripts/train_baseline.py --label-mode fixed --fixed-threshold 0.007
    python scripts/train_baseline.py --model hist --n-splits 6 --class-weight balanced

評価の要点（PoC の肝）:
- 分割は **日付境界の walk-forward**（同一日付の別銘柄が train/test をまたがない）。
- accuracy 単独で判断せず **macro-F1** も見る（FLAT 偏重で高精度に見える罠を避ける）。
- prev-direction baseline に対する **McNemar 検定**で accuracy の有意差を確認。
- fold 間 mean±std も併記（fold 相関のため厳密な検定ではない＝ヒューリスティック）。
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from flowsignal.eval.baselines import baseline_predictions, majority_class
from flowsignal.eval.metrics import (
    block_bootstrap_macro_f1,
    classification_metrics,
    mcnemar_test,
)
from flowsignal.eval.split import split_masks
from flowsignal.features.build import FEATURE_COLUMNS, load_and_build
from flowsignal.features.labels import LABEL_CLASSES, class_distribution
from flowsignal.models.baseline import RANDOM_SEED, feature_importances, make_model

_BASE_FROM_PRED = ["always_up", "random", "prev_direction"]  # baseline_predictions 由来
_BASELINES = ["always_majority", "prev_direction", "random", "always_up"]  # 報告順（強→弱）


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FlowSignal M2 ベースライン学習・評価")
    p.add_argument("--label-mode", choices=["vol", "fixed"], default="vol")
    p.add_argument("--k", type=float, default=0.5, help="ボラ連動の閾値係数")
    p.add_argument("--fixed-threshold", type=float, default=0.007)
    p.add_argument("--vol-window", type=int, default=20)
    p.add_argument("--model", choices=["lightgbm", "hist"], default="lightgbm")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="クラス不均衡対策（balanced で少数クラスを重み付け）",
    )
    p.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        help="macro-F1 の日付ブロック bootstrap の反復回数",
    )
    return p.parse_args()


def _fmt_pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def run_cv(df: pd.DataFrame, args: argparse.Namespace) -> dict:
    """walk-forward CV を回し、fold 別メトリクスと pooled OOS 予測を集計する。"""
    X = df[FEATURE_COLUMNS]
    y = df["label"]
    dates = df["date"]
    base_all = baseline_predictions(df, seed=args.seed)

    overrides = {}
    if args.class_weight == "balanced":
        overrides["class_weight"] = "balanced"

    fold_rows: list[dict] = []
    pooled_true: list[np.ndarray] = []
    pooled_model: list[np.ndarray] = []
    pooled_dates: list[np.ndarray] = []
    pooled_base: dict[str, list[np.ndarray]] = {b: [] for b in _BASELINES}
    last_model = None

    for fold, (tr, te) in enumerate(split_masks(df, n_splits=args.n_splits), start=1):
        model = make_model(args.model, seed=args.seed, **overrides)
        model.fit(X[tr], y[tr])
        pred = np.asarray(model.predict(X[te]))
        last_model = model

        y_te = y[te].to_numpy()
        m = classification_metrics(y_te, pred)
        row = {
            "fold": fold,
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "acc": m["accuracy"],
            "macro_f1": m["macro_f1"],
        }
        # baseline 予測（always_majority は **train 区間**の最頻クラス＝リーク無し）
        base_te = {b: base_all.loc[te, b].to_numpy() for b in _BASE_FROM_PRED}
        base_te["always_majority"] = np.full(int(te.sum()), majority_class(y[tr]))
        for b in _BASELINES:
            row[b] = float(np.mean(base_te[b] == y_te))
            pooled_base[b].append(base_te[b])
        fold_rows.append(row)

        pooled_true.append(y_te)
        pooled_model.append(pred)
        pooled_dates.append(dates[te].to_numpy())

    return {
        "folds": fold_rows,
        "y_true": np.concatenate(pooled_true),
        "y_model": np.concatenate(pooled_model),
        "y_base": {b: np.concatenate(v) for b, v in pooled_base.items()},
        "dates": np.concatenate(pooled_dates),
        "last_model": last_model,
    }


def _print_report(df: pd.DataFrame, res: dict, args: argparse.Namespace) -> None:
    folds = res["folds"]
    print(f"[cv] walk-forward {args.n_splits} folds（日付境界）/ model={args.model}"
          f" / class_weight={args.class_weight}\n")

    header = f"{'fold':>4} {'train':>6} {'test':>6} {'acc':>7} {'macroF1':>8} | " \
             f"{'maj':>6} {'prev':>6} {'rand':>6} {'up':>6}"
    print(header)
    print("-" * len(header))
    for r in folds:
        print(
            f"{r['fold']:>4} {r['n_train']:>6} {r['n_test']:>6} "
            f"{_fmt_pct(r['acc'])} {_fmt_pct(r['macro_f1'])} | "
            f"{_fmt_pct(r['always_majority'])} {_fmt_pct(r['prev_direction'])} "
            f"{_fmt_pct(r['random'])} {_fmt_pct(r['always_up'])}"
        )

    acc = np.array([r["acc"] for r in folds])
    f1 = np.array([r["macro_f1"] for r in folds])
    print(
        f"\n[fold mean±std] acc={acc.mean()*100:.1f}±{acc.std()*100:.1f}%  "
        f"macroF1={f1.mean()*100:.1f}±{f1.std()*100:.1f}%"
    )

    # --- pooled OOS ---
    yt, ym = res["y_true"], res["y_model"]
    m = classification_metrics(yt, ym)
    print(f"\n[pooled OOS] n={m['n']}  accuracy={_fmt_pct(m['accuracy'])}"
          f"  macro_f1={_fmt_pct(m['macro_f1'])}")
    print("  per-class (P / R / F1 / support):")
    for c in LABEL_CLASSES:
        pc = m["per_class"][c]
        print(f"    {c:>4}: {pc['precision']:.3f} / {pc['recall']:.3f} / "
              f"{pc['f1']:.3f} / {pc['support']}")
    print(f"  confusion (行=正解, 列=予測 {LABEL_CLASSES}):")
    for c, rowvals in zip(LABEL_CLASSES, m["confusion"]):
        print(f"    {c:>4} {rowvals}")

    # --- pooled 比較（accuracy と macro-F1 を併記＝指標で強弱が入れ替わる）---
    print("\n[pooled 比較] accuracy / macro-F1（accuracy では多数派が最強・macro-F1 では弱い）")
    print(f"    {'model':>16}: {_fmt_pct(m['accuracy'])} / {m['macro_f1']:.3f}")
    for b in _BASELINES:
        mb = classification_metrics(yt, res["y_base"][b])
        print(f"    {b:>16}: {_fmt_pct(mb['accuracy'])} / {mb['macro_f1']:.3f}")

    print("\n[McNemar] モデル vs ベースライン（accuracy 有意差, p<0.05 で有意）")
    for b in ("always_majority", "prev_direction", "always_up"):
        mc = mcnemar_test(yt, ym, res["y_base"][b])
        sig = "有意" if mc["pvalue"] < 0.05 else "有意差なし"
        winner = "model優位" if mc["n10"] > mc["n01"] else "baseline優位"
        print(f"    vs {b:>16}: stat={mc['statistic']:.2f} p={mc['pvalue']:.4g} "
              f"(n10={mc['n10']}, n01={mc['n01']}) -> {sig}・{winner}")

    # --- macro-F1 の日付ブロック bootstrap（McNemar が使えない本命指標の有意性）---
    pred_dict = {"model": ym, **{b: res["y_base"][b] for b in _BASELINES}}
    boot = block_bootstrap_macro_f1(
        yt, pred_dict, res["dates"], reference="model",
        n_boot=args.n_boot, seed=args.seed,
    )
    print(f"\n[bootstrap] macro-F1 95%CI（日次ブロック・n_boot={boot['n_boot']}）"
          " と model との差（CI 下限>0 なら有意）")
    for name in ["model"] + _BASELINES:
        lo, hi = boot["ci"][name]
        line = f"    {name:>16}: {boot['point'][name]:.3f} [{lo:.3f}, {hi:.3f}]"
        d = boot["diff_vs_reference"]["vs"].get(name)
        if d is not None:
            sig = "有意" if d["ci_low"] > 0 else "有意差なし"
            line += (f"  | Δ={d['diff']:+.3f} "
                     f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] {sig}")
        print(line)

    if res["last_model"] is not None and hasattr(res["last_model"], "feature_importances_"):
        imp = feature_importances(res["last_model"], FEATURE_COLUMNS).head(10)
        print("\n[feature importance top10]（最終 fold モデル）")
        for name, val in imp.items():
            print(f"    {name:>14}: {int(val)}")


def main() -> None:
    args = _parse_args()
    df = load_and_build(
        label_mode=args.label_mode,
        k=args.k,
        fixed_threshold=args.fixed_threshold,
        vol_window=args.vol_window,
    )
    dist = class_distribution(df["label"])
    total = int(dist.sum())
    print(f"[data] 学習テーブル {df.shape} / {df['code'].nunique()}銘柄 / "
          f"{df['date'].min().date()}..{df['date'].max().date()}")
    print("[labels] " + " | ".join(
        f"{c} {int(dist[c])} ({dist[c] / total * 100:.1f}%)" for c in LABEL_CLASSES
    ) + "\n")

    res = run_cv(df, args)
    _print_report(df, res, args)


if __name__ == "__main__":
    main()
