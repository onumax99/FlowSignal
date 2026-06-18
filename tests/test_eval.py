"""eval パッケージ（split / baselines / metrics）の単体テスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flowsignal.eval.baselines import baseline_predictions, majority_class
from flowsignal.eval.metrics import (
    block_bootstrap_macro_f1,
    classification_metrics,
    mcnemar_test,
)
from flowsignal.eval.split import split_masks, walk_forward_date_splits
from flowsignal.features.labels import LABEL_CLASSES


def _panel(n_dates=30, codes=("AAA", "BBB"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    frames = []
    for code in codes:
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "code": code,
                    "label": rng.choice(LABEL_CLASSES, size=n_dates),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# --- split -------------------------------------------------------------------


def test_walk_forward_temporal_order_and_no_overlap():
    df = _panel()
    folds = list(split_masks(df, n_splits=5))
    assert len(folds) == 5
    for train_mask, test_mask in folds:
        assert not (train_mask & test_mask).any()  # 同一行が両方に入らない
        assert train_mask.any() and test_mask.any()
        train_dates = df.loc[train_mask, "date"]
        test_dates = df.loc[test_mask, "date"]
        # 日付境界: 全 train 日付 < 全 test 日付
        assert train_dates.max() < test_dates.min()


def test_split_keeps_same_date_together():
    df = _panel()
    for _, test_mask in split_masks(df, n_splits=5):
        tmp = df.assign(in_test=test_mask)
        # どの日付も、その日の全銘柄が train か test のどちらか一方に入る。
        assert (tmp.groupby("date")["in_test"].nunique() <= 1).all()


def test_expanding_train_grows():
    df = _panel()
    train_sizes = [int(tr.sum()) for tr, _ in split_masks(df, n_splits=5)]
    assert train_sizes == sorted(train_sizes)
    assert train_sizes[0] < train_sizes[-1]


def test_too_few_dates_raises():
    df = _panel(n_dates=3)
    with pytest.raises(ValueError):
        list(walk_forward_date_splits(df["date"], n_splits=5))


# --- baselines ---------------------------------------------------------------


def test_always_up():
    df = _panel()
    preds = baseline_predictions(df)
    assert (preds["always_up"] == "UP").all()


def test_random_reproducible_and_in_classes():
    df = _panel()
    p1 = baseline_predictions(df, seed=99)["random"]
    p2 = baseline_predictions(df, seed=99)["random"]
    assert p1.tolist() == p2.tolist()
    assert set(p1.unique()).issubset(set(LABEL_CLASSES))


def test_prev_direction_uses_prior_label():
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=4),
            "code": "AAA",
            "label": ["UP", "DOWN", "FLAT", "UP"],
        }
    )
    prev = baseline_predictions(df)["prev_direction"].tolist()
    # 先頭は FLAT 穴埋め、以降は 1 つ前の実現ラベル。
    assert prev == ["FLAT", "UP", "DOWN", "FLAT"]


# --- metrics -----------------------------------------------------------------


def test_perfect_prediction_metrics():
    y = ["UP", "DOWN", "FLAT", "UP", "DOWN"]
    m = classification_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["n"] == 5


def test_known_accuracy():
    y_true = ["UP", "UP", "DOWN", "FLAT"]
    y_pred = ["UP", "DOWN", "DOWN", "FLAT"]  # 3/4 正解
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(0.75)
    assert np.array(m["confusion"]).sum() == 4


def test_mcnemar_identical_is_not_significant():
    y_true = ["UP"] * 50
    pred = ["UP"] * 25 + ["DOWN"] * 25
    res = mcnemar_test(y_true, pred, pred)
    assert res["n10"] == 0 and res["n01"] == 0
    assert res["pvalue"] == 1.0


def test_mcnemar_detects_clear_improvement():
    y_true = ["UP"] * 100
    model = ["UP"] * 100  # 全正解
    base = ["DOWN"] * 100  # 全不正解
    res = mcnemar_test(y_true, model, base)
    assert res["n10"] == 100 and res["n01"] == 0
    assert res["pvalue"] < 0.01


# --- majority baseline -------------------------------------------------------


def test_majority_class_picks_mode():
    assert majority_class(["FLAT", "FLAT", "UP", "DOWN", "FLAT"]) == "FLAT"


def test_majority_class_tie_breaks_by_class_order():
    # DOWN と UP が同数 → LABEL_CLASSES 先頭の DOWN を決定的に返す
    assert majority_class(["DOWN", "UP"]) == "DOWN"


def test_majority_class_ignores_nan():
    assert majority_class(pd.Series(["UP", "UP", np.nan, "FLAT"])) == "UP"


# --- macro-F1 block bootstrap ------------------------------------------------


def _boot_panel(n_days=40, codes=("AAA", "BBB"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    frames = [
        pd.DataFrame(
            {"date": dates, "code": c, "label": rng.choice(LABEL_CLASSES, size=n_days)}
        )
        for c in codes
    ]
    return pd.concat(frames, ignore_index=True)


def test_bootstrap_point_matches_sklearn_macro_f1():
    df = _boot_panel()
    yt = df["label"].to_numpy()
    yp = np.roll(yt, 1)  # 適当な予測（欠けるクラスがあっても sklearn と一致するか）
    boot = block_bootstrap_macro_f1(yt, {"a": yp}, df["date"].to_numpy(), n_boot=50, seed=1)
    assert boot["point"]["a"] == pytest.approx(
        classification_metrics(yt, yp)["macro_f1"]
    )


def test_bootstrap_reproducible_with_seed():
    df = _boot_panel()
    yt = df["label"].to_numpy()
    yp = np.roll(yt, 1)
    b1 = block_bootstrap_macro_f1(yt, {"a": yp}, df["date"].to_numpy(), n_boot=100, seed=7)
    b2 = block_bootstrap_macro_f1(yt, {"a": yp}, df["date"].to_numpy(), n_boot=100, seed=7)
    assert b1["ci"]["a"] == b2["ci"]["a"]


def test_bootstrap_ci_brackets_point():
    df = _boot_panel()
    yt = df["label"].to_numpy()
    yp = np.roll(yt, 1)
    boot = block_bootstrap_macro_f1(yt, {"a": yp}, df["date"].to_numpy(), n_boot=200, seed=3)
    lo, hi = boot["ci"]["a"]
    assert lo <= boot["point"]["a"] <= hi


def test_bootstrap_perfect_beats_bad_significantly():
    df = _boot_panel()
    yt = df["label"].to_numpy()
    perfect = yt.copy()
    bad = np.full(len(yt), "FLAT")  # 多数派一定予測（macro-F1 は低い）
    boot = block_bootstrap_macro_f1(
        yt,
        {"model": perfect, "bad": bad},
        df["date"].to_numpy(),
        reference="model",
        n_boot=300,
        seed=5,
    )
    d = boot["diff_vs_reference"]["vs"]["bad"]
    assert d["ci_low"] > 0  # 差の CI 下限が 0 超 = 有意に上
    assert d["p_one_sided"] == 0.0  # model ≤ bad となる resample は無い


def test_bootstrap_blocks_length_mismatch_raises():
    df = _boot_panel(n_days=10)
    yt = df["label"].to_numpy()
    with pytest.raises(ValueError):
        block_bootstrap_macro_f1(yt, {"a": yt}, df["date"].to_numpy()[:-1], n_boot=10)
