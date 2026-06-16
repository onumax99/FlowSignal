"""日次データ取得のエントリポイント（M1）。

使い方:
    python scripts/fetch_daily.py --days 365
    python scripts/fetch_daily.py --start 2024-01-01 --source jquants

デフォルトは yfinance（認証不要）。--source jquants で J-Quants を使用
（.env に認証情報が必要）。取得結果は data/raw/*.parquet と SQLite に保存する。
"""

from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from flowsignal import config
from flowsignal.data import market, news, storage
from flowsignal.data.prices import JQuantsClient, fetch_prices_yfinance


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FlowSignal 日次データ取得")
    p.add_argument("--start", help="取得開始日 YYYY-MM-DD（未指定なら --days を使用）")
    p.add_argument("--end", help="取得終了日 YYYY-MM-DD（既定: 今日）")
    p.add_argument("--days", type=int, default=365, help="--start 未指定時の遡及日数")
    p.add_argument(
        "--source",
        choices=["yfinance", "jquants"],
        default="yfinance",
        help="株価の取得元",
    )
    p.add_argument("--skip-news", action="store_true", help="ニュース取得をスキップ")
    return p.parse_args()


def _resolve_period(args: argparse.Namespace) -> tuple[str, str]:
    end = args.end or dt.date.today().isoformat()
    if args.start:
        start = args.start
    else:
        start = (dt.date.fromisoformat(end) - dt.timedelta(days=args.days)).isoformat()
    return start, end


def fetch_prices(args: argparse.Namespace, start: str, end: str) -> pd.DataFrame:
    universe = config.load_universe()
    if args.source == "jquants":
        client = JQuantsClient()
        if not client.available:
            raise SystemExit(
                "J-Quants の認証情報が未設定です。.env を設定するか --source yfinance を使用してください。"
            )
        frames = [
            client.fetch_daily_quotes(s.code, start, end) for s in universe.stocks
        ]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return fetch_prices_yfinance(universe.stock_yf_symbols, start, end)


def main() -> None:
    args = _parse_args()
    start, end = _resolve_period(args)
    config.ensure_dirs()
    print(f"[fetch] 期間 {start} - {end} / source={args.source}")

    prices = fetch_prices(args, start, end)
    storage.save_parquet(prices, "prices")
    print(f"[prices]  {len(prices):>6} 行 -> data/raw/prices.parquet")

    mkt = market.fetch_market(start, end)
    storage.save_parquet(mkt, "market")
    print(f"[market]  {len(mkt):>6} 行 -> data/raw/market.parquet")

    if not args.skip_news:
        articles = news.fetch_news()
        n = storage.upsert_news(articles)
        print(f"[news]    {n:>6} 件 -> SQLite(news)")

    print("[done] 取得完了")


if __name__ == "__main__":
    main()
