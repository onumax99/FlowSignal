"""株価 OHLCV の取得。

- yfinance: 認証不要で動作（PoC のデフォルト）
- J-Quants: 認証情報があれば利用（無料枠は当日値に遅延あり）

両者を切り替えられるよう、出力カラムを共通化する:
    columns = [date, code, open, high, low, close, volume]
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import requests

from flowsignal import config

_JQUANTS_BASE = "https://api.jquants.com/v1"


# --- yfinance ----------------------------------------------------------------


def fetch_prices_yfinance(
    yf_symbols: list[str],
    start: str | dt.date,
    end: str | dt.date | None = None,
) -> pd.DataFrame:
    """yfinance で日足 OHLCV を取得し、ロング形式の DataFrame を返す。

    返り値カラム: date, code, open, high, low, close, volume
    code は yfinance シンボル（例 "7203.T"）。
    """
    import yfinance as yf

    # auto_adjust=True: 分割・配当調整後の OHLC を返す（"Close" が調整後終値になる）。
    # 生 Close だと株式分割日に偽の極端リターンが出て、ラベル/テクニカル特徴量や
    # hv20 経由のボラ連動ラベル閾値を汚染する（→ docs/m2-evaluation.md 限界④）。
    # 注: 調整後終値は将来の分割/配当で過去系列が遡及調整され、配当ぶんはごく軽い look-ahead を伴う
    #     （リターンはトータルリターン化）。分割アーティファクト除去の効果が大きく PoC では許容する。
    raw = yf.download(
        tickers=yf_symbols,
        start=str(start),
        end=str(end) if end else None,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    frames: list[pd.DataFrame] = []
    for sym in yf_symbols:
        # 単一銘柄時は MultiIndex にならないため分岐
        sub = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
        if sub.empty:
            continue
        df = (
            sub.reset_index()
            .rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )[["date", "open", "high", "low", "close", "volume"]]
            .assign(code=sym)
        )
        frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=["date", "code", "open", "high", "low", "close", "volume"]
        )

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out[["date", "code", "open", "high", "low", "close", "volume"]]


# --- J-Quants ----------------------------------------------------------------


class JQuantsClient:
    """J-Quants API クライアント（認証 → 日足取得）。

    認証情報は config.jquants_credentials() から取得。
    フロー: auth_user(refresh token) -> auth_refresh(id token) -> daily_quotes
    """

    def __init__(self, mail: str | None = None, password: str | None = None):
        env_mail, env_pw = config.jquants_credentials()
        self.mail = mail or env_mail
        self.password = password or env_pw
        self._id_token: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.mail and self.password)

    def _authenticate(self) -> str:
        if not self.available:
            raise RuntimeError(
                "J-Quants の認証情報が未設定です（.env の JQUANTS_MAIL_ADDRESS / JQUANTS_PASSWORD）。"
            )
        r = requests.post(
            f"{_JQUANTS_BASE}/token/auth_user",
            json={"mailaddress": self.mail, "password": self.password},
            timeout=30,
        )
        r.raise_for_status()
        refresh_token = r.json()["refreshToken"]

        r = requests.post(
            f"{_JQUANTS_BASE}/token/auth_refresh",
            params={"refreshtoken": refresh_token},
            timeout=30,
        )
        r.raise_for_status()
        self._id_token = r.json()["idToken"]
        return self._id_token

    def _headers(self) -> dict[str, str]:
        token = self._id_token or self._authenticate()
        return {"Authorization": f"Bearer {token}"}

    def fetch_daily_quotes(
        self, code: str, start: str | dt.date, end: str | dt.date
    ) -> pd.DataFrame:
        """指定銘柄・期間の日足を取得（pagination 対応）。

        返り値カラム: date, code, open, high, low, close, volume
        """
        params = {"code": code, "from": str(start), "to": str(end)}
        records: list[dict] = []
        while True:
            r = requests.get(
                f"{_JQUANTS_BASE}/prices/daily_quotes",
                headers=self._headers(),
                params=params,
                timeout=30,
            )
            r.raise_for_status()
            payload = r.json()
            records.extend(payload.get("daily_quotes", []))
            key = payload.get("pagination_key")
            if not key:
                break
            params["pagination_key"] = key

        if not records:
            return pd.DataFrame(
                columns=["date", "code", "open", "high", "low", "close", "volume"]
            )

        # TODO(⑥): J-Quants も生 Close ではなく Adjustment*（分割調整後）列を使うべき。
        # 現状 M2 は yfinance(auto_adjust=True) 経由のため未対応。認証環境で検証してから切替える。
        df = pd.DataFrame(records).rename(
            columns={
                "Date": "date",
                "Code": "code",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df[["date", "code", "open", "high", "low", "close", "volume"]]
