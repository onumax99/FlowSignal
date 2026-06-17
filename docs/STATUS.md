# FlowSignal 開発ステータス / ハンドオフ

> このファイルは別セッション・別マシンへの引き継ぎ用。プロジェクトの現状・環境仕様・計画・タスク処理状況を一元化する。
> 関連: 要件定義 [requirements.md](requirements.md) / 概要 [../README.md](../README.md) / 予測アプローチ検討メモ [prediction-design.md](prediction-design.md)

- 最終更新: 2026-06-17
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

> **対応環境は Windows に一本化（PoC）。** 当初 macOS で開発を始めたが、lightgbm の OpenMP 依存（libomp）導入など macOS 固有のセットアップ負荷が大きいため、PoC では Windows 限定とする。macOS 固有の事情は §8 決定ログに履歴として残す。

| 項目 | 値 / 状況 |
|---|---|
| OS | Windows 11 Pro (10.0.26200) |
| Python | 3.13.12（`...\Programs\Python\Python313\python.exe`）。`py -3` では 3.14 も入っているが **3.13 を既定**とする |
| 仮想環境 | `.venv`（リポジトリ直下、git 管理外）。有効化: `.venv\Scripts\Activate.ps1`。**2026-06-17 に作成済み**（Python 3.13.12・`[ml,dev]` 導入） |
| パッケージ管理 | pip（`pip install -e .` / `pip install -e ".[ml]"`） |

### 主要パッケージ

依存は `pyproject.toml` 参照（core / `ml`=scikit-learn>=1.3・lightgbm>=4.0 / `app`=streamlit / `dev`=pytest）。

> ※ 2026-06-17 の構築で解決した主要版（py3.13）: **lightgbm 4.6.0・pandas 3.0.3・numpy 2.4.6・scikit-learn 1.9.0・scipy 1.17.1・pyarrow 24.0.0・yfinance 1.4.1**。pandas は旧 macOS venv（2.3.3）からメジャー更新したが、データ取得〜テクニカル特徴量は問題なく動作。版固定が必要になれば `pip freeze` で確定する。

### 環境メモ（Windows）

1. **lightgbm はそのまま動く想定（macOS の libomp 問題は無い）**
   - Windows 版 wheel は OpenMP ランタイムを同梱するため、`pip install -e ".[ml]"` 後に `import lightgbm` が**追加導入なしで通る**。**2026-06-17 に確認済み（lightgbm 4.6.0 / LGBMClassifier とも import OK）** → 前提が成立。
   - → これにより当初の「lightgbm 失敗時に HistGradientBoosting へ自動フォールバック」設計は**不要**。**lightgbm を単独の既定モデル**とする（§6.1 / §8）。
   - 補足: lightgbm・`HistGradientBoostingClassifier` とも NaN をネイティブに扱える（特徴量ウォームアップの先頭 NaN 対策に有利）。

2. **pandas-ta は不採用（テクニカル指標は自前実装）**
   - 主理由は **リーク防止のため計算を内製化したい**こと。加えて numpy 2.x 非互換（`from numpy import NaN`）の懸念も残る。
   - py3.13 では配布の入手自体はあり得るが、**自前実装の方針は維持**（`pyproject.toml` の `ml` extra から除外済み）。

---

## 3. リポジトリ / Git 状態

- ブランチ: `main`（`origin/main` を追跡）
- 最新コミット: `46e9b0f docs: 開発ステータス/ハンドオフ文書 (STATUS.md) を追加` 以降、本ハンドオフ時点の docs 改訂（README / STATUS / requirements / prediction-design）まで `origin/main` に反映済み。
- リモート（Windows 機）: `https://github.com/onumax99/FlowSignal.git`（**HTTPS**）。Windows は Git Credential Manager 経由の認証が標準。**初回 push 時に資格情報を確認**すること。
- **git 管理外**: `.venv/`、`.env`、`data/raw/`、`data/processed/`、`data/flowsignal.db`（`.gitignore` 済み）

> ※ 旧 macOS 機では SSH（ed25519 鍵を生成し GitHub 登録）で push していた（→ §8 決定ログ）。Windows 一本化に伴い、現用は HTTPS。

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
  features/
    technical.py              テクニカル指標13種を自前実装（後方参照のみ・M2 で追加）
scripts/fetch_daily.py        日次取得エントリポイント
tests/test_technical.py       technical.py の単体テスト（既知値＋先読み無し）
docs/requirements.md          要件定義書
docs/STATUS.md                本ファイル（ハンドオフ）
docs/prediction-design.md     予測アプローチ検討メモ（lead-lag / 較正 / クロスセクション、未確定）
pyproject.toml                依存定義（core / ml / app / dev extras。pytest 設定も同梱）
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
  baseline.py     lightgbm を既定モデル + seed固定（薄いファクトリで差し替え可。HistGradientBoosting は任意の代替）
src/flowsignal/eval/
  split.py        時系列分割（walk-forward / TimeSeriesSplit）※分割は必ず「日付境界」で
  baselines.py    always-up / random / prev-direction
  metrics.py      accuracy, macro-F1, per-class P/R, 混同行列, ベースライン比較, 有意性チェック
scripts/train_baseline.py   学習〜評価の通し実行（メトリクス表示、任意で predictions 保存）
```

### 6.2 設計メモ

- **ターゲット**: 翌営業日 close-to-close リターンの方向。閾値は PoC で調整（要件例 ±0.5〜1%、銘柄ボラ連動も検討）。
  - **クラス不均衡に注意**: 閾値が狭いと FLAT が痩せ、広いと FLAT 偏重になる。学習時に**クラス分布を必ず出力**し、不均衡が強ければ `class_weight='balanced'` 相当を検討する。
- **学習データ量**: 15銘柄 × 約242営業日 ≈ 3,630 行から、ウォームアップ（SMA20/RSI14 等の先頭 NaN）と test 配分を引くと標本が薄い。**M2 学習用は複数年（目安 3〜5年）に拡張**してメトリクスの分散・過学習を抑える（§6.3 の取得日数を参照）。
- **特徴量カテゴリ**（要件 8.1 準拠）:
  - テクニカル: リターン(1/5/10日)・移動平均乖離(SMA5/10/20)・RSI(14)・MACD(12,26,9)・ヒストリカルボラ(20日)・出来高変化
  - マーケット: 日経平均・TOPIX(ETF)・USDJPY・S&P500/NASDAQ（前日終値）・VIX のリターン/変化
  - カレンダー: 曜日・月初/月末・月（決算期の季節性の代理）
- **リーク防止の要点**: 特徴量は per-code で `shift` し当日以前のみ。ラベル付与時に未来行を落とす。
  - **米国指数の時点整合（最大のリーク源・規則を固定する）**: JST の取引日 t の特徴量に使う S&P500/NASDAQ は、**t の朝までに確定した直近の米クローズ（= t-1 の overnight 米セッション）**とする。この対応規則を `features/market.py` に明記してテストする。
  - **祝日 NaN**: market.parquet には国跨ぎ祝日由来の `close` NaN が存在（M1 で確認済み, 62行）→ **過去方向のみの前方補完（ffill）**で整合（未来参照を入れないこと）。
- **検証**: 時系列を尊重した分割でリーク防止（walk-forward 推奨）。PoC は全銘柄プールでの時系列分割から開始。
  - **分割は必ず「日付境界」で**: 全銘柄プールを行単位で split すると**同一日付の別銘柄が train/test をまたいでリーク**する。時刻 t の全銘柄は同一 fold に入るよう、カレンダー日付基準で割る。
- **評価**: 分類指標 + 必須のベースライン比較。
  - **「有意に上回るか」の判定**（PoC の肝）: walk-forward の fold ごとに精度を出し **mean±std** を見る、または prev-direction baseline に対する **McNemar 検定**で有意差を確認する。
    - 注: McNemar は accuracy（正誤の 2×2）に対する検定で macro-F1 は対象外。fold 間 mean±std は fold が相関するため厳密な検定ではない（PoC のヒューリスティックとして可）。
  - **confidence は未較正**: M2 の出力確信度は lightgbm の raw 確率（未較正）であり、そのままでは閾値判断に使えない。**確率較正（Platt/isotonic）は M3 で HOLD とセットで導入**（要件 §4・[prediction-design.md](prediction-design.md) ①）。
  - **M2 で意図的に対象外（M3/M4 へ送る）**: HOLD（棄権／確信度閾値, 要件 §4）、確率較正、AUC（要件 §9）、金融的評価（累積リターン等, M4 で本格化）。漏れではなく先送り。
- **predictions 保存の注意**: `predictions` の PK は `(date, code)` のため、ベースラインと後段モデルの予測が**上書き衝突**する。PoC では「ベースラインはメトリクスのみでDB保存しない」か、将来 `model`/`run` 列の追加を検討（当面は前者で可）。

### 6.3 セットアップ再現手順（クリーン環境）

```powershell
# Windows (PowerShell)
git clone https://github.com/onumax99/FlowSignal.git
cd FlowSignal
python -m venv .venv ; .venv\Scripts\Activate.ps1
pip install -e ".[ml]"                                   # M2 用（sklearn / lightgbm）
python -c "import lightgbm; print(lightgbm.__version__)" # Windows は追加導入なしで通る想定（初回のみ確認）
# 学習用は履歴を多めに（標本不足・過学習対策, §6.2「学習データ量」参照）
python scripts\fetch_daily.py --days 1825 --skip-news    # 約5年分の実データ取得（data\raw\*.parquet 生成）
```

---

## 7. タスク処理状況（チェックリスト）

### 完了
- [x] M1: データ取得モジュール一式（prices / market / news / storage / config）
- [x] M1: `scripts/fetch_daily.py` 実装
- [x] M1: 実データ通し確認（`--days 365 --skip-news`）
      → prices 3,630行（15銘柄×242営業日, close 欠損0）/ market 1,554行（6系列, 国跨ぎ祝日由来 NaN 62行）
- [x] M2 準備: pyproject の `ml` extra から pandas-ta を除外
- [x] M2 設計: モデルは **lightgbm 単独**で合意（自動フォールバックは廃止。薄いファクトリで差し替え可能性のみ残す → §8）
- [x] 初コミット作成 & `origin/main` へ push（旧 macOS 機で SSH。現用は Windows の HTTPS → §3/§8）
- [x] M2 環境構築（2026-06-17）: `.venv`(py3.13.12) 作成・`[ml,dev]` 導入・`import lightgbm` 確認・5年データ取得（prices 18,300行=15銘柄×各1,220 / market 7,812行=6系列, NaN は祝日由来で M2 で ffill 予定）
- [x] M2: `features/technical.py` テクニカル指標13種を自前実装（後方参照のみ＝リーク無し。実データ通し確認で NaN 件数＝ウォームアップ×15銘柄に一致）
- [x] M2: テクニカル指標の単体テスト 10 件（既知値照合＋**先読み無し**: 未来側を切っても過去の特徴量が完全一致する truncation invariance を assert）

### 未着手（M2 本体）
- [ ] `features/market.py` 指標・為替の特徴量化＋時点整合マージ
- [ ] `features/labels.py` 3クラスラベル（閾値方針の決定含む）
- [ ] `features/build.py` 学習テーブル生成（リーク防止）
- [ ] `models/baseline.py` lightgbm 既定＋薄いファクトリ＋seed固定（学習単位は **pooled で確定**。per-target は cross-stock 導入時=M3 で再検討 → §9）
- [ ] `eval/`（split[日付境界での walk-forward] / baselines / metrics[+有意性チェック]）
- [ ] `scripts/train_baseline.py` 通し実行
- [ ] M2 評価レポート（ベースライン比較＋有意性[fold mean±std / McNemar]のまとめ）

---

## 8. 主要な決定事項ログ

| 日付 | 決定 | 理由 |
|---|---|---|
| 2026-06-16 | **対応環境を Windows に一本化（macOS 対応は打ち切り）** | macOS は libomp/Homebrew 等のセットアップ負荷が大きい。Windows では lightgbm がそのまま動き構成が単純化するため |
| 2026-06-16 | モデルは **lightgbm 単独**（自動フォールバックは廃止） | Windows 一本化で libomp 問題が消失。HistGradientBoosting への自動フォールバックは不要に（薄いファクトリで差し替え可能性のみ残す） |
| 2026-06-16 | テクニカル指標は pandas-ta を使わず自前実装 | リーク制御を内製化したいのが主理由 / numpy 2.x 非互換の懸念 |
| 2026-06-16 | リモートは Windows 機で **HTTPS**（旧 macOS 機は SSH 化していた） | macOS 機では HTTPS 資格情報が無く SSH（ed25519）化。Windows 機は GCM 経由 HTTPS が標準で利用可 |
| 2026-06-16 | `.verify.txt` は git 管理外（コミットしない） | 検証用スクラッチファイルのため |

---

## 9. 未決事項（要件 13 章より、M2 で詰める）

- 対象銘柄リストの確定（現状 universe.yaml に大型15銘柄）
- 方向3クラスの閾値設定方針（固定 ±x% or ボラ連動）
- 学習単位: **M2 は pooled（全銘柄プール）で確定**。per-target（銘柄別）は cross-stock 特徴を入れる M3 で再検討（[prediction-design.md](prediction-design.md) §2④）
- **lead-lag / クロスセクション / 確率較正の M2 への取り込み可否**（[prediction-design.md](prediction-design.md) 提案、判断待ち）。現状 M2 はテクニカル単独に限定し、合意後に M2.5/M3 で反映する
