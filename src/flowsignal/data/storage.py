"""保存層: 時系列は parquet、ニュース等のレコードは SQLite。

PoC では軽量さを優先する。
- parquet: 株価・指標などの時系列データ（カラム指向で読み書きが速い）
- SQLite : ニュース記事・予測履歴などのメタ/レコード
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from flowsignal import config

# --- parquet -----------------------------------------------------------------


def save_parquet(df: pd.DataFrame, name: str, subdir: str = "raw") -> Path:
    """DataFrame を data/<subdir>/<name>.parquet に保存し、パスを返す。"""
    config.ensure_dirs()
    base = config.RAW_DIR if subdir == "raw" else config.PROCESSED_DIR
    path = base / f"{name}.parquet"
    df.to_parquet(path, index=True)
    return path


def load_parquet(name: str, subdir: str = "raw") -> pd.DataFrame:
    """data/<subdir>/<name>.parquet を読み込む。"""
    base = config.RAW_DIR if subdir == "raw" else config.PROCESSED_DIR
    return pd.read_parquet(base / f"{name}.parquet")


# --- SQLite ------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    id          TEXT PRIMARY KEY,   -- link の URL をキーに重複排除
    source      TEXT NOT NULL,
    published   TEXT,               -- ISO8601
    title       TEXT,
    summary     TEXT,
    link        TEXT,
    fetched_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    date        TEXT NOT NULL,      -- 予測実行日
    code        TEXT NOT NULL,      -- 証券コード
    label       TEXT,               -- UP / FLAT / DOWN / HOLD
    confidence  REAL,
    rationale   TEXT,               -- LLM による根拠（M4 以降）
    created_at  TEXT NOT NULL,
    PRIMARY KEY (date, code)
);
"""


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """SQLite 接続のコンテキストマネージャ。スキーマを初期化して返す。"""
    config.ensure_dirs()
    conn = sqlite3.connect(db_path or config.DB_PATH)
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_news(df: pd.DataFrame, db_path: Path | None = None) -> int:
    """ニュース記事を news テーブルに UPSERT。挿入/更新した件数を返す。"""
    if df.empty:
        return 0
    cols = ["id", "source", "published", "title", "summary", "link", "fetched_at"]
    rows = df[cols].itertuples(index=False, name=None)
    with connect(db_path) as conn:
        conn.executemany(
            f"INSERT OR REPLACE INTO news ({', '.join(cols)}) "
            f"VALUES ({', '.join(['?'] * len(cols))})",
            list(rows),
        )
    return len(df)
