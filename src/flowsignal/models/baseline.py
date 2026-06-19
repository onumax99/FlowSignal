"""ベースライン ML モデルの薄いファクトリ。

設計（STATUS §6.1 / §8）:
- **lightgbm を単独の既定モデル**とする（Windows wheel は OpenMP 同梱で追加導入不要）。
- 当初の「lightgbm 失敗時に HistGradientBoosting へ自動フォールバック」は廃止。
  ただし**差し替え可能性は薄いファクトリで残す**（`make_model("hist")`）。
- **再現性のため seed を固定**（RANDOM_SEED）。
- lightgbm・HistGradientBoosting とも NaN をネイティブに扱えるため、ウォームアップ
  由来の欠損を埋めずに渡せる。

返すのは sklearn 互換の分類器（fit / predict / predict_proba / classes_）。
ラベルは "UP"/"FLAT"/"DOWN" の文字列のまま渡せる（内部でエンコードされる）。
"""

from __future__ import annotations

import pandas as pd

RANDOM_SEED = 42

_LIGHTGBM_ALIASES = {"lightgbm", "lgbm", "lgb"}
_HIST_ALIASES = {"hist", "histgb", "hgb"}


def _lightgbm_defaults(seed: int) -> dict:
    return dict(
        objective="multiclass",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_samples=30,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def _hist_defaults(seed: int) -> dict:
    return dict(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=seed,
    )


def make_model(name: str = "lightgbm", *, seed: int = RANDOM_SEED, **overrides):
    """ベースライン分類器を生成する。

    Args:
        name: "lightgbm"（既定）または "hist"（HistGradientBoosting 代替）。
        seed: 乱数シード（再現性のため固定。既定 RANDOM_SEED=42）。
        **overrides: 既定ハイパーパラメータの上書き（例 class_weight="balanced"）。

    Returns:
        sklearn 互換の未学習分類器。
    """
    key = name.lower()
    if key in _LIGHTGBM_ALIASES:
        from lightgbm import LGBMClassifier

        params = _lightgbm_defaults(seed)
        params.update(overrides)
        return LGBMClassifier(**params)

    if key in _HIST_ALIASES:
        from sklearn.ensemble import HistGradientBoostingClassifier

        params = _hist_defaults(seed)
        params.update(overrides)
        return HistGradientBoostingClassifier(**params)

    raise ValueError(f"未知のモデル名: {name!r}（'lightgbm' または 'hist'）")


def _lightgbm_regressor_defaults(seed: int) -> dict:
    return dict(
        objective="regression",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_samples=30,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def make_regressor(name: str = "lightgbm", *, seed: int = RANDOM_SEED, **overrides):
    """回帰モデルを生成する（クロスセクションの相対リターン予測用）。

    分類器（make_model）と同じハイパラ思想・seed 固定。NaN ネイティブ対応。
    """
    key = name.lower()
    if key in _LIGHTGBM_ALIASES:
        from lightgbm import LGBMRegressor

        params = _lightgbm_regressor_defaults(seed)
        params.update(overrides)
        return LGBMRegressor(**params)

    if key in _HIST_ALIASES:
        from sklearn.ensemble import HistGradientBoostingRegressor

        params = _hist_defaults(seed)
        params.update(overrides)
        return HistGradientBoostingRegressor(**params)

    raise ValueError(f"未知のモデル名: {name!r}（'lightgbm' または 'hist'）")


def feature_importances(model, feature_names: list[str]) -> pd.Series:
    """学習済みモデルの特徴量重要度を降順 Series で返す（lightgbm 用）。"""
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("このモデルは feature_importances_ を持ちません。")
    return (
        pd.Series(model.feature_importances_, index=feature_names)
        .sort_values(ascending=False)
    )
