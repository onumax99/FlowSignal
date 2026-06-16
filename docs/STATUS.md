# FlowSignal 開発ステータス / ハンドオフ

> このファイルは別セッション・別マシンへの引き継ぎ用。プロジェクトの現状・環境仕様・計画・タスク処理状況を一元化する。
> 関連: 要件定義 [requirements.md](requirements.md) / 概要 [../README.md](../README.md)

- 最終更新: 2026-06-16
- 現在フェーズ: **M2（テクニカルのみのベースライン ML）着手前**（設計合意済み・コード未着手）
- リポジトリ: https://github.com/onumax99/FlowSignal

---

## 1. プロジェクト概要

日本株の**翌営業日の値動き方向**（UP / FLAT / DOWN ＋確信度）を、市場データの時系列 ML と
ニュースの LLM 分析のハイブリッドで予測する PoC。詳細は [requirements.md](requirements.md)。

PoC のゴールは「正確に当てる」ことではなく、**ベースライン（ランダム／常時上昇／前日同方向）を
有意に上回る予測の傾きを抽出できるか**の検証。

---

## 2. 開発環境・ツール仕様

| 項目 | 値 / 状況 |
|---|---|
| OS | macOS (Darwin 21.6.0, x86_64) |
| Python | 3.9.6（システム `/usr/bin/python3`） |
| 仮想環境 | `.venv`（リポジトリ直下、git 管理外） |
| パッケージ管理 | pip（`pip install -e .` / `pip install -e ".[ml]"`） |

### インストール済み主要パッケージ（venv 内、確認日 2026-06-16）

```
pandas==2.3.3        numpy==2.0.2        pyarrow==21.0.0
yfinance==1.2.0      feedparser==6.0.12  requests==2.32.5
python-dotenv==1.2.1 PyYAML==6.0.3
scikit-learn==1.6.1  lightgbm==4.6.0     scipy==1.13.1   joblib==1.5.3
```

### ⚠️ 環境上の重要な制約（M2 設計に直結）

1. **lightgbm は現状 import 不可**
   - macOS の OpenMP ランタイム `libomp.dylib` が未導入のため `import lightgbm` が `OSError` で失敗する。
   - Homebrew 自体が未導入。動かすには `Homebrew インストール → brew install libomp` が必要。
   - **方針（合意済み）**: モデル層を抽象化し、**lightgbm を優先・失敗時は sklearn の
     `HistGradientBoostingClassifier` に自動フォールバック**する。libomp を後で入れれば自動で lightgbm に戻る。
   - `HistGradientBoostingClassifier` は NaN をネイティブに扱え、特徴量重要度は `permutation_importance` で取得可。

2. **pandas-ta は不採用**
   - py3.9 向け配布が無く（利用可能版は 3.10+/3.12+ 要求）、かつ numpy 2.x 非互換（`from numpy import NaN`）。
   - `pyproject.toml` の `ml` extra から除外済み。**テクニカル指標は自前実装**する（リーク防止の観点でも内製が有利）。

3. **macOS システム Python は LibreSSL**
   - 実行時に `NotOpenSSLWarning`（urllib3 v2）が出るが無害。気になれば pyenv 等の OpenSSL ビルド Python へ。

---

## 3. リポジトリ / Git 状態

- ブランチ: `main`（`origin/main` を追跡）
- 最新コミット: `41ea189 M1: データ取得基盤と M2 準備`（push 済み）
- リモート: `git@github.com:onumax99/FlowSignal.git`（**SSH**。HTTPS は資格情報未設定で不可）
- SSH 鍵: `~/.ssh/id_ed25519`（ed25519・パスフレーズ無し、指紋 `SHA256:6zpyIt1...`）を新規生成し GitHub に登録済み。
  `github.com` のホスト鍵も公式指紋照合のうえ `known_hosts` 登録済み。→ このマシンからはそのまま push 可。
- **git 管理外**: `.venv/`、`.env`、`data/raw/`、`data/processed/`、`data/flowsignal.db`（`.gitignore` 済み）
- **未追跡で意図的に未コミット**: `.verify.txt`（検証スクラッチ。push 対象外）

---

## 4. ディレクトリ構成（現状）

```
config/universe.yaml          銘柄・指標・RSS の定義（株15 / 指標6 / RSS2）
src/flowsignal/
  config.py                   パス・.env・universe 読み込み（Universe/Stock/MarketSeries/NewsFeed）
  data/
    prices.py                 株価取得（yfinance / J-Quants）
    market.py                 指標・為替の取得（yfinance）
    news.py                   ニュース RSS 取得（feedparser）
    storage.py                保存層（parquet + SQLite スキーマ news/predictions）
scripts/fetch_daily.py        日次取得エントリポイント
docs/requirements.md          要件定義書
docs/STATUS.md                本ファイル（ハンドオフ）
pyproject.toml                依存定義（core / ml / app / dev extras）
```

### データ出力スキーマ

- `data/raw/prices.parquet`（ロング形式）: `date, code, open, high, low, close, volume`
  - `code` は yfinance シンボル（例 `7203.T`）
- `data/raw/market.parquet`（ロング形式）: `date, key, close`
  - `key` は内部キー（`nikkei225, topix_etf, usdjpy, sp500, nasdaq, vix`）
- `data/flowsignal.db`（SQLite）
  - `news(id, source, published, title, summary, link, fetched_at)`
  - `predictions(date, code, label, confidence, rationale, created_at)` ← M2 以降で利用

---

## 5. ロードマップ

| ID | 内容 | 完了条件 | 状態 |
|---|---|---|---|
| M1 | データ取得基盤 | 対象10〜30銘柄の株価＋ニュースを日次取得・保存できる | ✅ 完了（実データ通し確認済み） |
| M2 | ベースライン ML | テクニカルのみで方向予測し、ベースラインと比較評価できる | 🚧 設計合意済み・コード未着手 |
| M3 | LLM ニュース特徴量 | LLM スコアを追加し M2 比の改善幅を測定できる（PoC の肝） | 未着手 |
| M4 | バックテスト＋根拠説明 | 金融的評価指標を算出し、予測根拠を提示できる | 未着手 |
| M5 | ダッシュボード | Streamlit で銘柄別の予測・確信度・根拠を閲覧できる | 未着手 |

---

## 6. M2 実装計画（次セッションの作業）

> 完了条件: **テクニカルのみで翌営業日の3クラス方向を予測し、3種のベースラインと比較評価**できること。
> 必須要件: **データリーク防止**（特徴量は t 時点まで、ラベルは t→t+1）と **再現性**（seed 固定）。

### 6.1 提案モジュール構成

```
src/flowsignal/features/
  technical.py    テクニカル指標を自前実装（per stock, t時点まで）
  market.py       指標・為替を特徴量化し時点整合でマージ（米系は前日終値に注意）
  labels.py       翌日リターン → 3クラス（ボラ連動 or 固定閾値）
  build.py        上記を結合し学習用テーブルを生成（リーク防止の時点整合をここで担保）
src/flowsignal/models/
  baseline.py     モデル抽象化（lightgbm 優先 → HistGradientBoosting フォールバック）+ seed固定
src/flowsignal/eval/
  split.py        時系列分割（walk-forward / TimeSeriesSplit）
  baselines.py    always-up / random / prev-direction
  metrics.py      accuracy, macro-F1, per-class P/R, 混同行列, ベースライン比較
scripts/train_baseline.py   学習〜評価の通し実行（メトリクス表示、任意で predictions 保存）
```

### 6.2 設計メモ

- **ターゲット**: 翌営業日 close-to-close リターンの方向。閾値は PoC で調整（要件例 ±0.5〜1%、銘柄ボラ連動も検討）。
- **特徴量カテゴリ**（要件 8.1 準拠）:
  - テクニカル: リターン(1/5/10日)・移動平均乖離(SMA5/10/20)・RSI(14)・MACD(12,26,9)・ヒストリカルボラ(20日)・出来高変化
  - マーケット: 日経平均・TOPIX(ETF)・USDJPY・S&P500/NASDAQ（前日終値）・VIX のリターン/変化
  - カレンダー: 曜日・月初/月末・月
- **リーク防止の要点**: 特徴量は per-code で `shift` し当日以前のみ。米国指数は前日終値で整合。ラベル付与時に未来行を落とす。
  market.parquet には国跨ぎ祝日由来の `close` NaN が存在（M1 で確認済み）→ 整合/前方補完の処理が必要。
- **検証**: 時系列を尊重した分割でリーク防止（walk-forward 推奨）。PoC は全銘柄プールでの時系列分割から開始。
- **評価**: 分類指標 + 必須のベースライン比較。金融的評価（累積リターン等）は M4 で本格化。

### 6.3 セットアップ再現手順（クリーン環境）

```bash
git clone git@github.com:onumax99/FlowSignal.git && cd FlowSignal
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[ml]"          # M2 用（sklearn / lightgbm）
python scripts/fetch_daily.py --days 365 --skip-news   # 実データ取得（data/raw/*.parquet 生成）
# lightgbm を実際に使う場合のみ（任意）: brew install libomp
```

---

## 7. タスク処理状況（チェックリスト）

### 完了
- [x] M1: データ取得モジュール一式（prices / market / news / storage / config）
- [x] M1: `scripts/fetch_daily.py` 実装
- [x] M1: 実データ通し確認（`--days 365 --skip-news`）
      → prices 3,630行（15銘柄×242営業日, close 欠損0）/ market 1,554行（6系列, 国跨ぎ祝日由来 NaN 62行）
- [x] M2 準備: pyproject の `ml` extra から pandas-ta を除外
- [x] M2 設計: モデルは lightgbm 優先＋HistGradientBoosting 自動フォールバックで合意
- [x] 初コミット作成 & `origin/main` へ push（SSH 認証セットアップ含む）

### 未着手（M2 本体）
- [ ] `features/technical.py` テクニカル指標の自前実装
- [ ] `features/market.py` 指標・為替の特徴量化＋時点整合マージ
- [ ] `features/labels.py` 3クラスラベル（閾値方針の決定含む）
- [ ] `features/build.py` 学習テーブル生成（リーク防止）
- [ ] `models/baseline.py` モデル抽象化＋フォールバック＋seed固定
- [ ] `eval/`（split / baselines / metrics）
- [ ] `scripts/train_baseline.py` 通し実行
- [ ] M2 評価レポート（ベースライン比較結果のまとめ）

---

## 8. 主要な決定事項ログ

| 日付 | 決定 | 理由 |
|---|---|---|
| 2026-06-16 | テクニカル指標は pandas-ta を使わず自前実装 | py3.9 で配布無し / numpy 2.x 非互換 / リーク制御を内製化 |
| 2026-06-16 | モデルは lightgbm 優先＋sklearn HistGradientBoosting フォールバック | macOS に libomp/Homebrew 未導入。システム変更なしで動作させつつ、後から lightgbm へ自動移行 |
| 2026-06-16 | リモートを SSH 化（HTTPS は使わない） | HTTPS 資格情報が無く非対話環境で push 不可だったため。ed25519 鍵を生成し GitHub 登録 |
| 2026-06-16 | `.verify.txt` は git 管理外（コミットしない） | 検証用スクラッチファイルのため |

---

## 9. 未決事項（要件 13 章より、M2 で詰める）

- 対象銘柄リストの確定（現状 universe.yaml に大型15銘柄）
- 方向3クラスの閾値設定方針（固定 ±x% or ボラ連動）
- 学習単位（全銘柄プール vs 銘柄別モデル）
