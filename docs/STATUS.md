# FlowSignal 開発ステータス / ハンドオフ

> このファイルは別セッション・別マシンへの引き継ぎ用の**詳細ログ**（現状・環境仕様・計画・タスク処理状況）。
> 🆕 **新セッションはまず [handoff.md](handoff.md)（コールドスタート用の集約ブリーフ）を読むと速い。**
> 関連: 要件定義 [requirements.md](requirements.md) / 概要 [../README.md](../README.md) / 予測アプローチ検討メモ [prediction-design.md](prediction-design.md) / M2 結果 [m2-evaluation.md](m2-evaluation.md)

- 最終更新: 2026-06-17
- 現在フェーズ: **M2 完了 → 次は M2.5（評価の honest 化＋HOLD/較正/クロスセクション, API 不要）、その後 M3**（M2 評価レポート: [m2-evaluation.md](m2-evaluation.md)）
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
    market.py                 指標・為替の特徴量化＋時点整合（米系/FX は t-1 overnight・M2 で追加）
    labels.py                 翌日リターン→3クラス（既定=ボラ連動 k×HV20・M2 で追加）
    build.py                  特徴量＋ラベル結合→学習テーブル(24特徴量・M2 で追加)
    cross_section.py          日次デミーン目的変数＋日次 z-score 特徴量（M2.5 で追加）
  models/
    baseline.py               lightgbm 分類器ファクトリ＋make_regressor(回帰・M2.5)
  eval/
    split.py                  日付境界 walk-forward 分割（M2 で追加）
    baselines.py              always-up / random / prev-direction / majority_class（M2/M2.5）
    metrics.py                accuracy/macro-F1/per-class/混同行列＋McNemar＋macro-F1 日付ブロック bootstrap（M2/M2.5）
    calibration.py            Platt 多クラス較正（M2.5 で追加）
    hold.py                   HOLD: 確信度の閾値選択・coverage×macro-F1（M2.5 で追加）
    cross_section_metrics.py  rank IC / ロングショート / 日ブートストラップ（M2.5 で追加）
scripts/fetch_daily.py        日次取得エントリポイント
scripts/train_baseline.py     M2/honest化/HOLD の通し実行（--calibrate・M2/M2.5）
scripts/train_cross_section.py クロスセクション相対予測（M2.5 で追加）
tests/test_technical.py       technical.py の単体テスト（既知値＋先読み無し）
tests/test_market.py          market.py の単体テスト（時点整合・overnight・ffill）
tests/test_labels.py          labels.py の単体テスト（3クラス判定・σ後方参照）
tests/test_build.py           build.py の統合テスト（結合整合・市場ブロードキャスト）
tests/test_baseline.py        baseline.py の単体テスト（既定lgbm・NaN対応・再現性）
tests/test_eval.py            eval の単体テスト（分割・ベースライン・McNemar・bootstrap）
tests/test_hold.py            HOLD/Platt 較正の単体テスト（M2.5）
tests/test_cross_section.py   クロスセクション特徴量・rank IC/ロングショート（M2.5）
docs/handoff.md               新セッション向けコールドスタート・ブリーフ（集約版）
docs/requirements.md          要件定義書
docs/STATUS.md                本ファイル（詳細ログ）
docs/prediction-design.md     予測アプローチ検討メモ（lead-lag / 較正 / クロスセクション、未確定）
docs/m2-evaluation.md         M2 評価レポート（ベースライン比較＋有意性の結論・HOLD/クロスセクション §5/§6）
docs/m3-design.md             M3 設計書（データ方針/スコア仕様/プロンプト/コスト見積もり・課金前確定）
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
| M2 | ベースライン ML | テクニカルのみで方向予測し、ベースラインと比較評価できる | ✅ 完了（弱ベースラインに accuracy 有意。多数派/macro-F1 比較は M2.5 で追加・[評価レポート](m2-evaluation.md)） |
| M2.5 | 評価の honest 化＋低コスト改善（API 不要） | 多数派＋全ベースラインの macro-F1 併記・macro-F1 の bootstrap 有意性検定／HOLD＋確率較正／クロスセクション化 | ✅ 完了（honest 化／HOLD＋較正／クロスセクション。後 2 つは negative＝価格後処理では edge 無し） |
| M3 | LLM ニュース特徴量・適時開示 | LLM スコア（ニュース＋TDnet 開示）を追加し M2 比の改善幅を測定できる（PoC の肝） | 未着手 |
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
- [x] M2: `features/market.py` 指標・為替の特徴量化＋時点整合マージ（**米系/FX は ≤ t-1 overnight、JP 系は当日**。実データで「最終日 vix_level=16.2≠当日終値16.41」を確認しリーク無しを実証）
- [x] M2: `market.py` の単体テスト 6 件（JP 当日整合・米系 overnight・祝日 ffill・先読み無し）
- [x] M2: `features/labels.py` 3クラスラベル（**既定=ボラ連動 k×HV20, k=0.5**・fixed 切替可・σ は t 時点まで）。実データ分布 DOWN27.0/FLAT43.0/UP30.0%（adjusted close 再実行後）
- [x] M2: `labels.py` の単体テスト 9 件（固定閾値の既知値・最終日/ウォームアップ NaN・閾値の先読み無し）
- [x] M2: `features/build.py` 学習テーブル生成（24特徴量＝テクニカル13＋マーケット7＋カレンダー4。実データ **(17985, 29)**・label NaN 残0・残る特徴量 NaN は MACD ウォームアップのみで lightgbm が処理）
- [x] M2: `build.py` の統合テスト 6 件（スキーマ・市場ブロードキャスト・ラベル除外・カレンダー値域）
- [x] M2: `models/baseline.py` lightgbm 既定の薄いファクトリ＋seed 固定（hist 代替可・NaN ネイティブ対応・同一 seed で再現）
- [x] M2: `baseline.py` の単体テスト 8 件（既定 lgbm・未知名エラー・3クラス予測・NaN対応・再現性・重要度）
- [x] M2: `eval/`（split[日付境界 walk-forward] / baselines[3種] / metrics[accuracy・macro-F1・per-class・混同行列・McNemar]）＋単体テスト 12 件
- [x] M2: `scripts/train_baseline.py` 通し実行（pooled OOS n=15,000 で評価）
- [x] M2: 評価レポート [m2-evaluation.md](m2-evaluation.md)（**3ベースラインに accuracy 有意 / McNemar p≪0.001**。ただし FLAT 偏重で macro-F1 は 35〜37%＝方向 edge は弱い）
- [x] ⑥ 株価 adjusted close 化（`prices.py` を `auto_adjust=True`）＋5年データ再取得＋M2再実行（2026-06-18・分割/配当アーティファクト除去・既定 acc 41.0%/macroF1 34.9%・balanced 39.6%/37.2%・**結論不変**・pytest 全50件緑）
- [x] M2.5: 評価の honest 化＝always-majority ベースライン（`eval/baselines.majority_class`・fold ごとの train 最頻クラス＝リーク無し）＋全ベースラインの macro-F1 併記＋日付ブロック bootstrap（`eval/metrics.block_bootstrap_macro_f1`）。単体テスト8件追加（計58件緑）。**accuracy では多数派(43.4%)に有意に負け、macro-F1 では balanced(0.372)が全ベースラインに有意・既定(0.349)は prev-direction(0.345)と並ぶ**（2026-06-18）
- [x] M2.5: 確率較正(Platt)＋HOLD（確信度で棄権）＝`eval/calibration.py`(PlattCalibrator)/`eval/hold.py`＋単体テスト10件（計68件緑）。各 fold の train 末尾を validation に較正・閾値選択（リーク無し・`--calibrate`）。**結果は negative: 較正の argmax が FLAT に collapse し、HOLD は accuracy↑だが macro-F1 は ~0.22 で不変**（[m2-evaluation.md](m2-evaluation.md) §5）（2026-06-18）
- [x] M2.5: universe 15→30 銘柄に拡大（`config/universe.yaml`）＋5年再取得（prices 36,630 行）。survivorship bias あり（現存大型株から選定・PoC 許容）（2026-06-19）
- [x] M2.5: クロスセクション（相対）予測＝`features/cross_section.py`（日次デミーン目的＋日次 z-score）/`models/baseline.make_regressor`/`eval/cross_section_metrics.py`（rank IC・ロングショート・日ブートストラップ）/`scripts/train_cross_section.py`＋単体テスト8件（計76件緑）。**結果は negative: mean IC=-0.0024≈ランダム・95%CI が 0 跨ぎ非有意、ロングショート -4.7bp/Sharpe -0.65**（[m2-evaluation.md](m2-evaluation.md) §6）（2026-06-19）- [x] M3 step ①（着手・課金前スキャフォールド 2026-06-19）: `config.anthropic_api_key()`＋`anthropic` 依存（`llm` extra）＋`scripts/measure_tokens.py`（count_tokens で §6 を実測・各モデルの cache 最小プレフィックス検証つき・**無料**）＋単体テスト3件（計 79 件緑）。**count_tokens の実行＝鍵設定＋予算登録の後**（[m3-design.md](m3-design.md) §6.3 / §5.1）。新発見: 既定の固定プレフィックスは Opus 4.8 の cache 最小（4,096 tok）を下回る見込み→§6 のコスト上振れに注意。

> **M2＋M2.5 完了**（2026-06-19）。pytest 全 76 件グリーン。**M2.5 の 3 手法（honest 化／HOLD＋較正／クロスセクション）完了・後 2 つは negative＝価格後処理では edge 無し**。universe 30 銘柄。次は M3（LLM ニュース特徴量・初の Claude API 課金）。

---

## 8. 主要な決定事項ログ

| 日付 | 決定 | 理由 |
|---|---|---|
| 2026-06-16 | **対応環境を Windows に一本化（macOS 対応は打ち切り）** | macOS は libomp/Homebrew 等のセットアップ負荷が大きい。Windows では lightgbm がそのまま動き構成が単純化するため |
| 2026-06-16 | モデルは **lightgbm 単独**（自動フォールバックは廃止） | Windows 一本化で libomp 問題が消失。HistGradientBoosting への自動フォールバックは不要に（薄いファクトリで差し替え可能性のみ残す） |
| 2026-06-16 | テクニカル指標は pandas-ta を使わず自前実装 | リーク制御を内製化したいのが主理由 / numpy 2.x 非互換の懸念 |
| 2026-06-16 | リモートは Windows 機で **HTTPS**（旧 macOS 機は SSH 化していた） | macOS 機では HTTPS 資格情報が無く SSH（ed25519）化。Windows 機は GCM 経由 HTTPS が標準で利用可 |
| 2026-06-16 | `.verify.txt` は git 管理外（コミットしない） | 検証用スクラッチファイルのため |
| 2026-06-18 | 株価を **adjusted close**（yfinance `auto_adjust=True`）に切替 | 生 Close は分割・配当でリターンが跳ね、偽の極端リターン・hv20 経由のラベル閾値の歪みで M2 を汚染するため（再実行で結論は不変だがデータ健全化） |

---

## 9. 未決事項（要件 13 章より）

- 🚨 **【M3 ブロッカー】データ可用性 × LLM hindsight**: `data/news.py` は RSS のみで過去アーカイブを返さず、M1/M2 は `--skip-news` で実行済み＝5 年分のニュース履歴が無い。M3 を M2 と同じ walk-forward で評価するには過去ニュースが必要だが、**forward 限定評価** と **TDnet 等の履歴ソース確保** は**リークの観点で等価でない**: `claude-opus-4-8`（カットオフ 2026-01）に 2021〜2025 の履歴を採点させると hindsight が残るため、構造的にリーク無しなのは**カットオフ後の forward 区間だけ**。→ **本筋は forward-only**、履歴は特徴量開発・デバッグ用と割り切る（[handoff.md](handoff.md) §5・§6 / [requirements.md](requirements.md) §13）。
- ✅ **評価の honest 化（M2.5 で完了 2026-06-18）**: `eval/baselines.py` に always-majority(FLAT, fold ごとの train 最頻クラス)を追加、全ベースラインの macro-F1 を併記、`eval/metrics.block_bootstrap_macro_f1`（日付ブロック bootstrap）で CI・差を算出。結論＝**accuracy では多数派(43.4%)に有意に負け、macro-F1 では balanced(0.372)が全ベースラインに有意・既定(0.349)は prev-direction(0.345)と並ぶ**（[m2-evaluation.md](m2-evaluation.md)「限界・追加検証」）。
- **適時開示(TDnet)の取り込み**: 要件 §6・[prediction-design.md](prediction-design.md) §4③ が差別化の本命とするが、現状 universe.yaml / news.py に TDnet ソースが無い。`data/disclosure.py` 相当を **M3 の必須タスク**として追加する。
- 対象銘柄リストの確定（現状 universe.yaml に大型15銘柄）。クロスセクション/lead-lag は 1 日の断面が薄いと成立しないため、**銘柄数拡大（30+）は M2.5 クロスセクション化の前提条件**（前倒し必須・API 不要）。
- ✅ **株価の分割・配当調整（対応済み 2026-06-18）**: `data/prices.py` を yfinance `auto_adjust=True`（adjusted close）に切替え、5 年データ再取得＋ M2 再実行。生 `Close` 時代の分割アーティファクト（偽の極端リターン・hv20 経由のラベル閾値の歪み）を除去（集計指標の変化は小・結論不変）。J-Quants 経路は未対応（`prices.py` の TODO）（[m2-evaluation.md](m2-evaluation.md) 限界④）。
- ~~方向3クラスの閾値設定方針~~ → **決定済み（2026-06-17）: 既定はボラ連動 k×HV20（k=0.5）。固定 ±x% も切替可**（`labels.py`）。k は分布を見て調整可
- 学習単位: **M2 は pooled（全銘柄プール）で確定**。per-target（銘柄別）は cross-stock 特徴を入れる M3 で再検討（[prediction-design.md](prediction-design.md) §2④）
- **M2.5 の 3 手法は完了・HOLD/較正・クロスセクションは negative**: 較正の argmax が FLAT collapse で macro-F1 不変（[m2-evaluation.md](m2-evaluation.md) §5）、クロスセクション相対予測も mean IC≈0・非有意（§6）。価格後処理では edge 無し→新情報(M3)が本命。lead-lag は②③の枠内の一特徴として後続で検討。
