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

from flowsignal.eval.baselines import baseline_predictions
from flowsignal.eval.metrics import classification_metrics, mcnemar_test
from flowsignal.eval.split import split_masks
from flowsignal.features.build import FEATURE_COLUMNS, load_and_build
from flowsignal.features.labels import LABEL_CLASSES, class_distribution
from flowsignal.models.baseline import RANDOM_SEED, feature_importances, make_model

_BASELINES = ["always_up", "random", "prev_direction"]


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
    return p.parse_args()


def _fmt_pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def run_cv(df: pd.DataFrame, args: argparse.Namespace) -> dict:
    """walk-forward CV を回し、fold 別メトリクスと pooled OOS 予測を集計する。"""
    X = df[FEATURE_COLUMNS]
    y = df["label"]
    base_all = baseline_predictions(df, seed=args.seed)

    overrides = {}
    if args.class_weight == "balanced":
        overrides["class_weight"] = "balanced"

    fold_rows: list[dict] = []
    pooled_true: list[np.ndarray] = []
    pooled_model: list[np.ndarray] = []
    pooled_base: dict[str, list[np.ndarray]] = {b: [] for b in _BASELINES}
    last_model = None

    for fold, (tr, te) in enumerate(split_masks(df, n_splits=args.n_splits), start=1):
        model = make_model(args.model, seed=args.seed, **overrides)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        last_model = model

        y_te = y[te].to_numpy()
        row = {
            "fold": fold,
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "acc": classification_metrics(y_te, pred)["accuracy"],
            "macro_f1": classification_metrics(y_te, pred)["macro_f1"],
        }
        for b in _BASELINES:
            bp = base_all.loc[te, b].to_numpy()
            row[b] = float(np.mean(bp == y_te))
            pooled_base[b].append(bp)
        fold_rows.append(row)

        pooled_true.append(y_te)
        pooled_model.append(np.asarray(pred))

    return {
        "folds": fold_rows,
        "y_true": np.concatenate(pooled_true),
        "y_model": np.concatenate(pooled_model),
        "y_base": {b: np.concatenate(v) for b, v in pooled_base.items()},
        "last_model": last_model,
    }


def _print_report(df: pd.DataFrame, res: dict, args: argparse.Namespace) -> None:
    folds = res["folds"]
    print(f"[cv] walk-forward {args.n_splits} folds（日付境界）/ model={args.model}"
          f" / class_weight={args.class_weight}\n")

    header = f"{'fold':>4} {'train':>6} {'test':>6} {'acc':>7} {'macroF1':>8} | " \
             f"{'up':>6} {'rand':>6} {'prev':>6}"
    print(header)
    print("-" * len(header))
    for r in folds:
        print(
            f"{r['fold']:>4} {r['n_train']:>6} {r['n_test']:>6} "
            f"{_fmt_pct(r['acc'])} {_fmt_pct(r['macro_f1'])} | "
            f"{_fmt_pct(r['always_up'])} {_fmt_pct(r['random'])} {_fmt_pct(r['prev_direction'])}"
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

    print("\n[pooled baseline accuracy]")
    for b in _BASELINES:
        acc_b = float(np.mean(res["y_base"][b] == yt))
        print(f"    {b:>14}: {_fmt_pct(acc_b)}")

    print("\n[McNemar] モデル vs ベースライン（accuracy 有意差, p<0.05 で有意）")
    for b in ("prev_direction", "always_up"):
        mc = mcnemar_test(yt, ym, res["y_base"][b])
        sig = "有意" if mc["pvalue"] < 0.05 else "有意差なし"
        print(f"    vs {b:>14}: stat={mc['statistic']:.2f} p={mc['pvalue']:.4g} "
              f"(n10={mc['n10']}, n01={mc['n01']}) -> {sig}")

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
