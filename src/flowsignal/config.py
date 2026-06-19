"""設定とパスの一元管理。

- プロジェクトのディレクトリ構成（data/raw, data/processed など）
- `.env` からの認証情報読み込み
- `config/universe.yaml` の読み込み（銘柄・指標・ニュースフィード定義）
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

# リポジトリルート（このファイルから src/flowsignal/config.py -> 2 つ上）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "flowsignal.db"

CONFIG_DIR = PROJECT_ROOT / "config"
UNIVERSE_PATH = CONFIG_DIR / "universe.yaml"

# .env を一度だけ読み込む
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Stock:
    code: str
    yf_symbol: str
    name: str


@dataclass(frozen=True)
class MarketSeries:
    key: str
    yf_symbol: str
    name: str


@dataclass(frozen=True)
class NewsFeed:
    source: str
    url: str


@dataclass(frozen=True)
class Universe:
    stocks: list[Stock] = field(default_factory=list)
    market: list[MarketSeries] = field(default_factory=list)
    news_feeds: list[NewsFeed] = field(default_factory=list)

    @property
    def stock_yf_symbols(self) -> list[str]:
        return [s.yf_symbol for s in self.stocks]

    @property
    def market_yf_symbols(self) -> list[str]:
        return [m.yf_symbol for m in self.market]


@lru_cache(maxsize=1)
def load_universe(path: Path | None = None) -> Universe:
    """universe.yaml を読み込んで Universe を返す（キャッシュ付き）。"""
    path = path or UNIVERSE_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Universe(
        stocks=[Stock(**s) for s in raw.get("stocks", [])],
        market=[MarketSeries(**m) for m in raw.get("market", [])],
        news_feeds=[NewsFeed(**n) for n in raw.get("news_feeds", [])],
    )


def ensure_dirs() -> None:
    """データ用ディレクトリを作成する（存在すれば何もしない）。"""
    for d in (RAW_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)


def jquants_credentials() -> tuple[str | None, str | None]:
    """J-Quants のログイン情報を環境変数から取得。未設定なら (None, None)。"""
    return (
        os.getenv("JQUANTS_MAIL_ADDRESS") or None,
        os.getenv("JQUANTS_PASSWORD") or None,
    )


def anthropic_api_key() -> str | None:
    """Claude API キーを環境変数から取得。未設定なら None。

    M3（LLM ニュース/開示スコア化）で使用。`anthropic.Anthropic()` は
    同じ `ANTHROPIC_API_KEY` を自動で読むため、本ヘルパは主に事前の有無チェック用。
    """
    return os.getenv("ANTHROPIC_API_KEY") or None
