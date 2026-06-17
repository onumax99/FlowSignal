"""翌営業日リターンを 3 クラス（UP / FLAT / DOWN）へ変換するラベル生成。

定義:
- ターゲットは **翌営業日 close-to-close リターン** r_fwd(t) = close(t+1)/close(t) - 1。
- 閾値方式（切替可能・既定は "vol"）:
    - "vol"  : 閾値 = k × σ_t。σ_t は直近ボラ（HV20 相当＝日次リターンの
               vol_window 日標準偏差, **t 時点まで**）。銘柄・時期に適応し FLAT 偏重を抑える。
    - "fixed": 閾値 = 固定 ±x%（全銘柄共通）。
- ラベル: r_fwd > +閾値 → UP / r_fwd < -閾値 → DOWN / それ以外 → FLAT。

リーク防止:
- 閾値に使う σ_t は **t 以前のみ**（rolling, 後方参照）で計算する。未来は参照しない。
- 未来を見るのは r_fwd（= 予測対象そのもの）の close(t+1) だけ。各銘柄の最終日は
  t+1 が無いためラベルは NaN（build.py 側で学習対象から落ちる）。
- σ_t のウォームアップ（先頭 vol_window 行）はラベル NaN。

クラス不均衡対策として、生成後は必ずクラス分布を確認すること（class_distribution）。

入出力はロング形式:
    入力 : date, code, close
    出力 : date, code, ret_fwd, threshold, label
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_CLASSES: list[str] = ["DOWN", "FLAT", "UP"]
LABEL_COLUMN = "label"

_REQUIRED_COLUMNS = ("date", "code", "close")


def _labels_one(
    g: pd.DataFrame, *, mode: str, k: float, fixed_threshold: float, vol_window: int
) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    close = g["close"].astype("float64")

    ret_fwd = close.shift(-1) / close - 1.0  # 翌日リターン（ラベル対象）
    daily_ret = close.pct_change(fill_method=None)  # 当日までの日次リターン

    if mode == "vol":
        sigma = daily_ret.rolling(vol_window, min_periods=vol_window).std()
        threshold = k * sigma
    elif mode == "fixed":
        threshold = pd.Series(float(fixed_threshold), index=close.index)
    else:
        raise ValueError(f"未知の mode: {mode!r}（'vol' または 'fixed'）")

    up = ret_fwd > threshold
    down = ret_fwd < -threshold
    label = pd.Series(
        np.select([up, down], ["UP", "DOWN"], default="FLAT"),
        index=close.index,
        dtype=object,
    )
    # r_fwd / threshold が未定義の行はラベルも未定義（NaN）にする。
    valid = ret_fwd.notna() & threshold.notna()
    label = label.where(valid, other=np.nan)

    return pd.DataFrame(
        {
            "date": g["date"],
            "code": g["code"],
            "ret_fwd": ret_fwd,
            "threshold": threshold,
            "label": label,
        }
    )


def compute_labels(
    prices: pd.DataFrame,
    *,
    mode: str = "vol",
    k: float = 0.5,
    fixed_threshold: float = 0.007,
    vol_window: int = 20,
) -> pd.DataFrame:
    """ロング形式の株価から銘柄ごとの 3 クラスラベルを生成する。

    Args:
        prices: date, code, close を含むロング形式 DataFrame。
        mode: "vol"（既定, 閾値 = k×σ_t）または "fixed"（閾値 = fixed_threshold）。
        k: vol モードの閾値係数（既定 0.5）。
        fixed_threshold: fixed モードの固定閾値（既定 0.007 = ±0.7%）。
        vol_window: σ の計算窓（既定 20 = HV20 と整合）。

    Returns:
        date, code, ret_fwd, threshold, label を持つロング形式 DataFrame
        （code, date 昇順）。ウォームアップ・各銘柄最終日は label が NaN。
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in prices.columns]
    if missing:
        raise ValueError(f"prices に必要な列がありません: {missing}")

    if prices.empty:
        return pd.DataFrame(
            columns=["date", "code", "ret_fwd", "threshold", "label"]
        )

    frames = [
        _labels_one(
            g, mode=mode, k=k, fixed_threshold=fixed_threshold, vol_window=vol_window
        )
        for _, g in prices.groupby("code", sort=False)
    ]
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["code", "date"]).reset_index(drop=True)


def class_distribution(labels: pd.Series) -> pd.Series:
    """ラベルのクラス分布（NaN 除外、UP/FLAT/DOWN の件数）を返す。"""
    counts = labels.dropna().value_counts()
    return counts.reindex(LABEL_CLASSES, fill_value=0)
