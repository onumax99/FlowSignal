# FlowSignal

各種銘柄・市場の動きと時事ニュースから関連性を抽出し、**日本株の翌営業日の値動き方向**（UP / FLAT / DOWN ＋確信度）を予測する PoC。市場データの時系列ML と、ニュースの LLM 分析を組み合わせるハイブリッド方式。

- 🆕 引き継ぎブリーフ（新セッションはまずこれ）: [docs/handoff.md](docs/handoff.md)
- 要件定義: [docs/requirements.md](docs/requirements.md)
- 開発ステータス / 詳細ログ: [docs/STATUS.md](docs/STATUS.md)
- M2 評価レポート: [docs/m2-evaluation.md](docs/m2-evaluation.md)
- 現在のフェーズ: **M2 完了 → 次は M2.5（HOLD＋較正・クロスセクション・評価の honest 化, API 不要）**。その後 M3（LLM ニュース特徴量）

> ⚠️ 個人投資の意思決定補助を目的とした実験的ツールです。予測の正確性・収益を保証しません。投資は自己責任で行ってください。

## セットアップ

> 対応環境は **Windows（PoC）**。Python 3.13 での動作を前提とします。

```powershell
# Windows (PowerShell)
python -m venv .venv ; .venv\Scripts\Activate.ps1

pip install -e .              # 取得基盤のみ
# pip install -e ".[ml,app]"  # M2 以降（モデル・ダッシュボード）

copy .env.example .env        # 必要に応じて J-Quants 等の認証情報を設定
```

> Windows の日本語環境（コンソールが cp932）で文字化け・`UnicodeEncodeError` が出る場合は、環境変数 `PYTHONUTF8=1` を設定するか Windows Terminal の利用を推奨します。

## 使い方

### M1: データ取得

```powershell
# yfinance（認証不要）で過去1年を取得
python scripts/fetch_daily.py --days 365

# M2 学習用に約5年分（ニュースは不要なら --skip-news）
python scripts/fetch_daily.py --days 1825 --skip-news

# J-Quants を使う場合（.env に認証情報が必要）
python scripts/fetch_daily.py --start 2024-01-01 --source jquants
```

取得結果:
- `data/raw/prices.parquet` … 対象銘柄の日足 OHLCV
- `data/raw/market.parquet` … 指標・為替（日経平均・ドル円・S&P500・VIX など）
- `data/flowsignal.db` (SQLite) … ニュース記事（`news` テーブル）、予測履歴（`predictions`）

対象銘柄・指標・ニュースフィードは [config/universe.yaml](config/universe.yaml) で定義。

### M2: ベースライン学習・評価

```powershell
# テクニカルのみで翌日方向を予測し、3ベースラインと比較（日付境界 walk-forward）
python scripts/train_baseline.py

# クラス不均衡対策・固定閾値などの切替も可能
python scripts/train_baseline.py --class-weight balanced --label-mode fixed
```

結果の解釈は [docs/m2-evaluation.md](docs/m2-evaluation.md) を参照。

## ディレクトリ構成

```
config/universe.yaml      銘柄・指標・RSS の定義
src/flowsignal/
  config.py               パス・設定・universe 読み込み
  data/                   取得層（prices / market / news / storage）
  features/               特徴量（technical / market / labels / build）
  models/baseline.py      lightgbm 既定の薄いファクトリ
  eval/                   評価（split / baselines / metrics）
scripts/fetch_daily.py    日次取得エントリポイント
scripts/train_baseline.py M2 学習〜評価の通し実行
tests/                    pytest（特徴量のリーク制御・評価ロジック）
docs/                     requirements / STATUS / prediction-design / m2-evaluation
```

## ロードマップ

| ID | 内容 | 状態 |
|---|---|---|
| M1 | データ取得基盤 | ✅ 完了 |
| M2 | テクニカルのみのベースラインML（方向予測＋評価） | ✅ 完了（弱ベースラインに accuracy 有意。多数派/macro-F1 比較は M2.5 で追加） |
| M2.5 | HOLD＋確率較正・クロスセクション化・評価の honest 化（API 不要） | 🚧 次の本命 |
| M3 | LLM ニュース特徴量・適時開示(TDnet)の追加（改善幅を測定） | 未着手 |
| M4 | バックテスト＋根拠説明 | 未着手 |
| M5 | Streamlit ダッシュボード | 未着手 |
