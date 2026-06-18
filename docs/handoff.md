# FlowSignal 引き継ぎブリーフ（新セッション向け）

> **新しいセッションが最初に読むコールドスタート用の要約。** これまでの結果・実装済み仕様・
> 今後の計画を 1 枚に集約する。粒度の細かいログや背景は下記の各 doc を正準とする。
>
> - 詳細ステータス / 決定ログ: [STATUS.md](STATUS.md)
> - 要件定義: [requirements.md](requirements.md)
> - M2 評価結果の詳細: [m2-evaluation.md](m2-evaluation.md)
> - 将来設計（精度向上の方針）: [prediction-design.md](prediction-design.md)

- 最終更新: 2026-06-17
- リポジトリ: https://github.com/onumax99/FlowSignal （`main` 直コミット運用・HTTPS）

---

## 0. 30 秒サマリ

- **何を作っているか**: 日本株の**翌営業日の値動き方向**（UP / FLAT / DOWN ＋確信度）を、
  市場データの時系列 ML と LLM ニュース分析のハイブリッドで予測する PoC。対応環境は **Windows 一本化**。
- **現在地**: **M1（データ取得）・M2（テクニカルのみベースライン ML）完了。次は M2.5（HOLD＋較正・
  クロスセクション・評価の honest 化, API 不要）、その後 M3（LLM ニュース特徴量）。**
- **M2 の結論（正直に）**: **弱ベースライン（always-up / random / prev-direction）に accuracy で有意**
  （McNemar p≪0.001）。ただし FLAT 偏重で **macro-F1 35〜37%＝方向当ての真の edge は弱い**。
  さらに **多数派（always-FLAT≈43%）ベースラインは未掲載で、モデル 41.2% はこれを上回らない見込み**
  （= 「有意」は弱ベースライン限定）。重要度はマーケット/overnight 系が支配的。
- **健全性**: `pytest` 全 50 件グリーン。`main` は `origin/main` と同期済み。リーク防止はテストで担保。
- **ゴールの考え方**: 「正確に当てる」ではなく「**ベースラインを有意に上回る傾きを、リークなく抽出できるか**」。

---

## 1. すぐ動かす（環境再現）

> **対応環境は Windows（PoC）。** Python 3.13 前提。日本語コンソール出力には `PYTHONUTF8=1` が必要
> （未設定だと `UnicodeEncodeError`）。`.venv` と `data/` は git 管理外なので、別マシンでは再構築する。

```powershell
# 1) 仮想環境（Python 3.13）
py -3.13 -m venv .venv ; .venv\Scripts\Activate.ps1
pip install -e ".[ml,dev]"
python -c "import lightgbm; print(lightgbm.__version__)"   # 追加導入なしで通る想定

# 2) データ取得（M2 学習用に約5年・ニュース不要）
$env:PYTHONUTF8=1
python scripts\fetch_daily.py --days 1825 --skip-news

# 3) M2 ベースライン学習・評価
python scripts\train_baseline.py
python scripts\train_baseline.py --class-weight balanced   # FLAT偏重の緩和を見る

# 4) テスト
python -m pytest
```

確定済み主要版（py3.13・2026-06-17）: lightgbm 4.6.0 / pandas **3.0.3** / numpy 2.4.6 /
scikit-learn 1.9.0 / scipy 1.17.1 / yfinance 1.4.1。

---

## 2. アーキテクチャ & データ

```
config/universe.yaml          銘柄15 / 指標6 / RSS2 の定義
src/flowsignal/
  config.py                   パス・.env・universe 読み込み
  data/                       取得層: prices(yf/J-Quants) / market / news(RSS) / storage(parquet+SQLite)
  features/                   technical / market / labels / build
  models/baseline.py          lightgbm 既定の薄いファクトリ
  eval/                       split(日付境界WF) / baselines(3種) / metrics(+McNemar)
scripts/fetch_daily.py        日次データ取得
scripts/train_baseline.py     M2 学習〜評価の通し実行
tests/                        pytest（リーク制御・評価ロジック）
```

データスキーマ（`data/raw`・`data/flowsignal.db`、いずれも git 管理外）:
- `prices.parquet`（ロング）: `date, code, open, high, low, close, volume`（code は yf シンボル例 `7203.T`）
- `market.parquet`（ロング）: `date, key, close`（key: `nikkei225, topix_etf, usdjpy, sp500, nasdaq, vix`）
- SQLite: `news(id, source, published, title, summary, link, fetched_at)` /
  `predictions(date, code, label, confidence, rationale, created_at)`（PK=(date,code)）

---

## 3. 実装済み仕様（M2 まで）

### リーク防止の約束事（全モジュール共通・最重要）
- 特徴量は**後方参照のみ**（diff / rolling / ewm）。各日付 t の値は t 以前のみに依存。
- **米国指数・FX は ≤ t-1 の overnight** に整合（JP 大引け t には当日 US 終値は未確定）。JP 系は当日終値。
- ラベルが未来を見るのは r_fwd の `close(t+1)` だけ。各銘柄の最終日はラベル NaN。
- 検証は**日付境界の walk-forward**（同一日付の別銘柄が train/test をまたがない）。
- これらは各 `tests/test_*.py` の「先読み無し（truncation invariance）」等で assert 済み。

### features/technical.py — テクニカル13特徴量（`TECHNICAL_FEATURES`）
`ret_1, ret_5, ret_10 / sma5_dev, sma10_dev, sma20_dev / rsi14 / macd, macd_signal, macd_hist / hv20 / vol_chg, vol_ratio20`。
入出力ロング。ウォームアップ NaN は埋めない（lightgbm が処理）。

### features/market.py — マーケット7特徴量（`MARKET_FEATURES`）
`ret_nikkei, ret_topix, ret_usdjpy, ret_sp500, ret_nasdaq, chg_vix, vix_level`。
`compute_market_features(market, trading_dates)`。米系/FX は ≤ t-1 整合、祝日 NaN は過去方向 ffill。

### features/labels.py — 3クラスラベル
`compute_labels(prices, mode="vol", k=0.5, fixed_threshold=0.007, vol_window=20)`。
**既定はボラ連動: 閾値 = k×σ_t（σ_t は HV20 相当・t 時点まで）**。`mode="fixed"` で固定 ±x% に切替可。
出力 `date, code, ret_fwd, threshold, label`。`class_distribution()` で分布確認。

### features/build.py — 学習テーブル（24特徴量・`FEATURE_COLUMNS`）
テクニカル13＋マーケット7＋カレンダー4（`dow, month, is_month_start, is_month_end`）を (date, code) で結合。
`build_dataset(prices, market, ...)` / `load_and_build(...)`。既定でラベル NaN 行を除外。実データ **(17985, 29)**。

### models/baseline.py — モデルファクトリ
`make_model("lightgbm", seed=42, **overrides)`。**lightgbm 単独が既定**（自動フォールバックは廃止）、
`make_model("hist")` で HistGradientBoosting 代替。NaN ネイティブ対応・同一 seed で再現。`feature_importances()` 付き。

### eval/ — 評価
- `split.py`: `split_masks(df, n_splits=5)` = 日付境界 expanding walk-forward。
- `baselines.py`: `baseline_predictions(df)` = always-up / random / prev-direction。
- `metrics.py`: `classification_metrics()`（accuracy・macro-F1・per-class・混同行列）、`mcnemar_test()`。

---

## 4. M2 結果（要点・詳細は [m2-evaluation.md](m2-evaluation.md)）

pooled OOS n=15,000、日付境界 5-fold、ボラ連動ラベル k=0.5。

| 設定 | accuracy | macro-F1 |
|---|---|---|
| 既定（class_weight=none） | **41.2%** | 35.1% |
| balanced | 39.0% | **36.9%** |
| ベースライン prev-direction | 36.3% | – |
| ベースライン random / always-up | 33.1% / 29.6% | – |

- McNemar（既定）: vs prev-direction **p=2.3e-19**、vs always-up **p=1.5e-78** → いずれも有意。
  ただし **McNemar は accuracy 専用**で、その accuracy は FLAT 偏重で水増しされている指標である点に注意。
- 重要度上位は `vix_level, ret_usdjpy, ret_topix, ret_nikkei, chg_vix, 米指数リターン`（マーケット系が支配的、
  自前テクニカルは下位）。→ テクニカル作り込みは収穫逓減という想定を裏づけ。
- ⚠️ **評価の既知の限界（M2.5 で対応）**: ①多数派（always-FLAT≈43%）ベースライン未掲載＝モデル 41.2% は
  accuracy でこれを上回らない見込み ②ベースラインの macro-F1 未算出＝本命指標での勝ちは未検証
  （指標で強弱が逆転: always-FLAT の macro-F1 ~0.20、prev-direction ~0.33-0.36 が macro-F1 の本命対戦相手）
  ③macro-F1 の有意性検定（**日次/時間ブロックの block bootstrap**、5 fold の再標本化ではない）が未実施。詳細・対応 TODO は [m2-evaluation.md](m2-evaluation.md) の「限界・追加検証」。

---

## 5. これからの計画

### 次の本命: M2.5（HOLD＋較正・クロスセクション・評価の honest 化）★API 不要・最優先
prediction-design §4 の「①②最優先」と M2 の評価限界（§4 ⚠️）に直接効く、課金ゼロの工程。
**データもコストも重い M3 より先に、ここで無償の改善を積む。** M2.5 が macro-F1 を動かせるか確認できれば、LLM 課金前に多くを学べる。
- **評価の honest 化（M2 の限界対応）**: `eval/baselines.py` に **always-majority(FLAT)** を追加し、
  全ベースラインの **macro-F1 を併記**。本命指標 macro-F1 は **fold ブロック bootstrap で CI** を出して有意性を判定
  （McNemar は accuracy 専用なので不可）。→ [m2-evaluation.md](m2-evaluation.md) を更新。
- **HOLD（確信度で棄権）＋確率較正（Platt/isotonic, 小標本は Platt 既定）**: 高確信日だけ予測し、
  **カバレッジ×macro-F1 のトレードオフ曲線**で評価（棄権すると covered set が FLAT に偏り accuracy は自動で上がるため・クラス構成も併記）。
  較正・閾値・k・class_weight は train/validation で凍結（test で選ばない）。
- **クロスセクション（相対）予測**: 市場共通要因(β)を外し「同日内でどれが相対的に強いか」を当てる。
  銘柄固有シグナルが学習しやすい。**評価は 3 クラス accuracy/macro-F1 でなく日次 rank IC / top-k−bottom-k スプレッド**。
  15 銘柄では断面が薄く rank IC がノイジーなので **universe 拡大（15→30+）はクロスセクション化の前提条件**（前倒し必須・API 不要）。

### その後の本命: M3（LLM ニュース特徴量・適時開示）
要件 §8.2 / prediction-design §3・§4③ を踏まえた具体化。**ここで初めて Claude API が課金対象**（M1/M2/M2.5 は無料）。

- 🚨 **着手前の最重要確認（データ実在性 × LLM hindsight）**: 現状の取得層 `data/news.py` は **RSS のみ**で、RSS は直近記事しか返さず
  **過去アーカイブを返さない**。M1/M2 は `--skip-news` で走っており、5 年分のニュース履歴は無い。
  → **「M2 と同じ walk-forward で M3 の改善幅を測る」には過去ニュースが必要**。設計前に次のどちらかを決めるが、
  **(a) と (b) はリークの観点で等価でない**:
  (a) 評価を「今から日次収集していく forward 区間」に限定、または (b) 履歴の取れるソースを確保
  （**TDnet 適時開示は履歴取得が可能**で、§4③ の本命でもある）。
  **(b) で履歴を得ても `claude-opus-4-8`（カットオフ 2026-01）に 2021〜2025 を採点させると hindsight が残る**ため、
  構造的にリーク無しなのは**カットオフ後の forward 区間だけ**。→ **PoC の本筋は (a) forward-only**、(b) 履歴は
  特徴量パイプラインの開発・デバッグ用（backtest 数値はリーク前提の参考値）と役割を分ける。
- **成功基準を事前登録（pre-register）**: M2 で edge が薄い＋ hindsight リスクがあるため、着手前に合否を定義する。
  例「**材料のある日**条件で macro-F1 または rank IC の block-bootstrap CI 下限が prev-direction を上回る」。後から良い指標を選ばない。
- **TDnet 取り込みを M3 タスクに明示**: 要件 §6・prediction-design §4③ は **決算・開示(TDnet)の event-driven を
  差別化の本命**と位置づけるが、現状 universe.yaml / news.py に TDnet ソースが無い。`data/disclosure.py` 相当の
  **TDnet 取り込みを M3 の必須タスク**として追加し、汎用 RSS より優先する。
- **やること**: news/開示を LLM でスコア化 → ①登場企業・テーマ抽出と対象銘柄マッピング
  ②感情(ポジ/ネガ)・インパクト・関連度の数値化 → 特徴量化し、build テーブルに**時点整合**で結合 →
  **M2 比の改善幅（macro-F1/accuracy）を同じ walk-forward ハーネスで測定**。
  **pooled だけでなく「材料のある日」条件付き**でも測る（イベント効果は pooled だと希釈される）。
- **新規モジュール案**: `features/news_llm.py`（LLM スコア化）、`data/disclosure.py`（TDnet）、
  プロンプト/few-shot 定義、スコアの parquet 保存。
- **リーク注意（ニュース特有）**:
  - 記事の `published` が JP 大引け t **以前**のものだけを row t に使う（t の終値後に出たニュースは t+1 行へ）。
    market.py の時点整合と同じ思想で `news_llm.py` に固定＋テスト。
  - 🆕 **LLM 自身の hindsight リーク**: Claude（知識カットオフ 2026-01）に過去記事を採点させると、
    **その後の結末を「知っている」未来情報がスコアに混入**しうる。対策: プロンプトで「記事テキストのみで判断・
    後知恵/外部知識を使うな」と明示、**LLM に方向（上下）を予測させず感情/関連度/新規性の定量化に限定**、
    可能なら日付を伏せて採点。
- **モデル**: 学習するのは引き続き LightGBM のみ。LLM は**推論専用**（ファインチューニングしない）。
  既定 `claude-opus-4-8`、コスト重視 `claude-sonnet-4-6`、大量スコア化は `claude-haiku-4-5`。
- **コスト最適化**: ① Batch API（50%引・日次バッチと好相性）② prompt caching（指示文＋few-shot を固定プレフィックス化）
  ③ モデル階層化。**見積もりは2種類を分ける**: backtest は「過去コーパス一括採点（記事数×トークンの一回コスト）」、
  運用は「日次 forward コスト（≈15銘柄/日 $0.5〜1.5）」。実数は着手時に `messages.count_tokens` で確定。

### その先（prediction-design 参照・合意後）
- **lead-lag**: 銘柄間 lead-lag（市場中立化残差）。per-target モデルや銘柄数拡大とセット。②③の枠の「一特徴」として。
- **M4**: バックテスト＋根拠説明（金融的評価: 累積リターン・シャープ・最大DD）。
- **M5**: Streamlit ダッシュボード（`[app]` extra に streamlit 済み）。

---

## 6. 未決事項 / 引き継ぎ時の注意

- 🚨 **【最重要・M3 ブロッカー】ニュース履歴が無い**: `data/news.py` は RSS のみで過去アーカイブを返さない。
  M3 の「M2 と同じ walk-forward で改善幅測定」には過去ニュース/開示が要る。**設計前にデータ実在性を決着**
  （forward 限定評価 or TDnet 等の履歴ソース確保）。→ §5 M3 の「着手前の最重要確認」。
- ⚠️ **M2 評価の限界（M2.5 で対応）**: 多数派(always-FLAT≈43%)ベースライン未掲載／ベースラインの macro-F1 未算出／
  macro-F1 の有意性検定未実施。「accuracy で有意」は弱ベースライン限定。→ [m2-evaluation.md](m2-evaluation.md)「限界・追加検証」。
- **閾値 k** は分布を見て調整可（現状 k=0.5 で FLAT 43%）。`--label-mode fixed` でも比較できる。
- **対象銘柄**は大型15銘柄（universe.yaml）。クロスセクション/ lead-lag を狙うなら銘柄数拡大が効く（M2.5 で前倒し可・API 不要）。
- **predictions テーブルの PK=(date,code)** はベースラインと本モデルで上書き衝突しうる。DB 保存を本格化するなら
  `model`/`run` 列の追加を検討（現状 train_baseline は DB 保存せずメトリクス出力のみ）。
- **J-Quants 無料枠の遅延**が評価に与える影響は未確認（現状 yfinance で取得）。
- 🆕 **株価が分割・配当未調整（M2 数値に影響し得る）**: `data/prices.py` は yfinance `auto_adjust=False`＋生 `Close`。
  分割で偽の極端リターンが出るため **adjusted close へ切替＋ M2 再実行**が要対応（[STATUS.md](STATUS.md) §9・[m2-evaluation.md](m2-evaluation.md) 限界）。
- **class_weight**: 既定 none。balanced で macro-F1↑/accuracy↓のトレードオフ。
- pandas は **3.0 系**（旧 macOS venv は 2.3.3）。取得〜特徴量は動作確認済みだが、新規コードで 3.0 破壊的変更に注意。

---

## 7. リポジトリ / Git 運用

- `main` に直コミット＋ `origin/main` へ push する運用（個人 PoC）。実装ステップ完了ごとに 1 コミット。
- 認証は **HTTPS**（Windows は Git Credential Manager）。Git 識別子はこのリポジトリ限定で
  `YukiOnuma <onumax99@gmail.com>`。
- git 管理外: `.venv/`, `.env`, `data/`（`.gitignore` 済み）。
