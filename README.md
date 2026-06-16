# FlowSignal

各種銘柄・市場の動きと時事ニュースから関連性を抽出し、**日本株の翌営業日の値動き方向**（UP / FLAT / DOWN ＋確信度）を予測する PoC。市場データの時系列ML と、ニュースの LLM 分析を組み合わせるハイブリッド方式。

- 要件定義: [docs/requirements.md](docs/requirements.md)
- 現在のフェーズ: **M1（データ取得基盤）**

> ⚠️ 個人投資の意思決定補助を目的とした実験的ツールです。予測の正確性・収益を保証しません。投資は自己責任で行ってください。

## セットアップ

```bash
# macOS / Linux
python -m venv .venv && source .venv/bin/activate

# Windows (PowerShell)
#   python -m venv .venv ; .venv\Scripts\Activate.ps1
# Windows (cmd)
#   python -m venv .venv && .venv\Scripts\activate.bat

pip install -e .            # 取得基盤のみ
# pip install -e ".[ml,app]"  # M2 以降（モデル・ダッシュボード）

cp .env.example .env        # 必要に応じて J-Quants 等の認証情報を設定
# Windows: copy .env.example .env
```

> Windows の日本語環境（コンソールが cp932）で文字化け・`UnicodeEncodeError` が出る場合は、環境変数 `PYTHONUTF8=1` を設定するか Windows Terminal の利用を推奨します。

## 使い方（M1: データ取得）

```bash
# yfinance（認証不要）で過去1年を取得
python scripts/fetch_daily.py --days 365

# J-Quants を使う場合（.env に認証情報が必要）
python scripts/fetch_daily.py --start 2024-01-01 --source jquants
```

取得結果:
- `data/raw/prices.parquet` … 対象銘柄の日足 OHLCV
- `data/raw/market.parquet` … 指標・為替（日経平均・ドル円・S&P500・VIX など）
- `data/flowsignal.db` (SQLite) … ニュース記事（`news` テーブル）、予測履歴（`predictions`）

対象銘柄・指標・ニュースフィードは [config/universe.yaml](config/universe.yaml) で定義。

## ディレクトリ構成

```
config/universe.yaml      銘柄・指標・RSS の定義
src/flowsignal/
  config.py               パス・設定・universe 読み込み
  data/
    prices.py             株価取得（yfinance / J-Quants）
    market.py             指標・為替の取得
    news.py               ニュース RSS 取得
    storage.py            保存層（parquet + SQLite）
scripts/fetch_daily.py    日次取得エントリポイント
docs/requirements.md      要件定義書
```

## ロードマップ

| | 内容 | 状態 |
|---|---|---|
| M1 | データ取得基盤 | 🚧 進行中 |
| M2 | テクニカルのみのベースラインML（方向予測＋評価） | 未着手 |
| M3 | LLM ニュース特徴量の追加（改善幅を測定） | 未着手 |
| M4 | バックテスト＋根拠説明 | 未着手 |
| M5 | Streamlit ダッシュボード | 未着手 |
