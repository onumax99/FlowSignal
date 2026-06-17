"""baseline.py（モデルファクトリ）の単体テスト。

- 既定は lightgbm・seed 固定。未知名は ValueError。
- 文字列ラベルをそのまま学習でき、3 クラスを予測できる。
- NaN を含む特徴量でも学習・予測できる（ネイティブ NaN 対応）。
- 同一 seed で予測が再現する（再現性）。
"""

from __future__ import annotations

import numpy as np
import pytest

from flowsignal.models.baseline import RANDOM_SEED, feature_importances, make_model


def _toy_classification(n=300, seed=0, with_nan=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5))
    score = 1.5 * x[:, 0] - 1.0 * x[:, 1] + rng.normal(0, 0.5, n)
    y = np.where(score > 0.7, "UP", np.where(score < -0.7, "DOWN", "FLAT"))
    if with_nan:
        x[rng.integers(0, n, 20), 2] = np.nan
    return x, y


def test_make_model_defaults_to_lightgbm_with_seed():
    from lightgbm import LGBMClassifier

    m = make_model()
    assert isinstance(m, LGBMClassifier)
    assert m.get_params()["random_state"] == RANDOM_SEED


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        make_model("xgboost")


def test_overrides_applied():
    m = make_model("lightgbm", n_estimators=50)
    assert m.get_params()["n_estimators"] == 50


def test_fit_predict_three_classes():
    x, y = _toy_classification()
    model = make_model(n_estimators=80)
    model.fit(x, y)
    pred = model.predict(x)
    assert set(np.unique(pred)).issubset({"UP", "FLAT", "DOWN"})
    # 学習データ上はランダム(0.33)より明確に良いはず。
    assert (pred == y).mean() > 0.5
    proba = model.predict_proba(x)
    assert proba.shape == (len(y), len(model.classes_))


def test_handles_nan_features():
    x, y = _toy_classification(with_nan=True)
    model = make_model(n_estimators=50)
    model.fit(x, y)  # NaN があっても例外なく学習できる
    assert len(model.predict(x)) == len(y)


def test_reproducible_with_same_seed():
    x, y = _toy_classification()
    p1 = make_model(seed=123, n_jobs=1, n_estimators=80).fit(x, y).predict_proba(x)
    p2 = make_model(seed=123, n_jobs=1, n_estimators=80).fit(x, y).predict_proba(x)
    np.testing.assert_allclose(p1, p2)


def test_feature_importances_sorted():
    x, y = _toy_classification()
    names = [f"f{i}" for i in range(5)]
    model = make_model(n_estimators=50).fit(x, y)
    imp = feature_importances(model, names)
    assert list(imp.index[:1])[0] in names
    assert imp.is_monotonic_decreasing


def test_hist_alternative_available():
    from sklearn.ensemble import HistGradientBoostingClassifier

    m = make_model("hist")
    assert isinstance(m, HistGradientBoostingClassifier)
