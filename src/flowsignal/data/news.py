"""ニュース取得（RSS）。

universe.yaml の news_feeds を feedparser で取得し、共通スキーマに正規化する。
LLM による銘柄マッピング・感情スコア化は M3 で別モジュールに実装する。

出力カラム: id, source, published, title, summary, link, fetched_at
"""

from __future__ import annotations

import datetime as dt
import hashlib

import pandas as pd

from flowsignal import config


def _entry_id(link: str, title: str) -> str:
    """重複排除キー。link があれば link、無ければ title のハッシュ。"""
    base = link or title or ""
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _to_iso(entry) -> str | None:
    """feedparser の published_parsed (time.struct_time) を ISO8601 に変換。"""
    parsed = getattr(entry, "published_parsed", None) or getattr(
        entry, "updated_parsed", None
    )
    if parsed is None:
        return None
    return dt.datetime(*parsed[:6]).isoformat()


def fetch_news() -> pd.DataFrame:
    """全 RSS フィードを取得し正規化した DataFrame を返す。"""
    import feedparser

    universe = config.load_universe()
    fetched_at = dt.datetime.now().isoformat()
    rows: list[dict] = []

    for feed in universe.news_feeds:
        parsed = feedparser.parse(feed.url)
        for e in parsed.entries:
            link = getattr(e, "link", "")
            title = getattr(e, "title", "")
            rows.append(
                {
                    "id": _entry_id(link, title),
                    "source": feed.source,
                    "published": _to_iso(e),
                    "title": title,
                    "summary": getattr(e, "summary", ""),
                    "link": link,
                    "fetched_at": fetched_at,
                }
            )

    cols = ["id", "source", "published", "title", "summary", "link", "fetched_at"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].drop_duplicates(subset="id")
