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
- **現在地**: **M1（データ取得）・M2（テクニカルのみベースライン ML）完了。次は M3（LLM ニュース特徴量）。**
- **M2 の結論（正直に）**: 3 ベースラインに **accuracy で統計的有意**（McNemar p≪0.001）だが、
  FLAT 偏重で **macro-F1 35〜37%＝方向当ての真の edge は弱い**。重要度はマーケット/overnight 系が支配的。
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
- 重要度上位は `vix_level, ret_usdjpy, ret_topix, ret_nikkei, chg_vix, 米指数リターン`（マーケット系が支配的、
  自前テクニカルは下位）。→ テクニカル作り込みは収穫逓減という想定を裏づけ。

---

## 5. これからの計画

### 次セッションの本命: M3（LLM ニュース特徴量）
要件 §8.2 / prediction-design §3・§4 を踏まえた具体化。**ここで初めて Claude API が課金対象**（M1/M2 は無料）。

- **やること**: SQLite `news` の記事を LLM でスコア化 → ①登場企業・テーマ抽出と対象銘柄マッピング
  ②感情(ポジ/ネガ)・インパクト・関連度の数値化 → 特徴量化し、build テーブルに**時点整合**で結合 →
  **M2 比の改善幅（macro-F1/accuracy）を同じ walk-forward ハーネスで測定**。
- **新規モジュール案**: `features/news_llm.py`（LLM スコア化）、プロンプト/few-shot 定義、スコアの parquet 保存。
- **リーク注意（ニュース特有）**: 記事の `published` が JP 大引け t **以前**のものだけを row t に使う
  （t の終値後に出たニュースは t+1 行へ）。market.py の時点整合と同じ思想で `news_llm.py` に固定＋テスト。
- **モデル**: 学習するのは引き続き LightGBM のみ。LLM は**推論専用**（ファインチューニングしない）。
  既定 `claude-opus-4-8`、コスト重視 `claude-sonnet-4-6`、大量スコア化は `claude-haiku-4-5`。
- **コスト最適化**: ① Batch API（50%引・日次バッチと好相性）② prompt caching（指示文＋few-shot を固定プレフィックス化）
  ③ モデル階層化。実数は着手時に `messages.count_tokens` で見積もる。

### 低コストで先に効く候補（API 不要・M2.5 として先行可）
prediction-design §4 の「①最優先」。M2 の弱点（FLAT 偏重・確信度が未較正）に直接効く。
- **HOLD（確信度で棄権）＋確率較正（Platt/isotonic, 小標本は Platt 既定）**: 高確信日だけ予測し、
  **カバレッジ×精度のトレードオフ曲線**で評価。較正・閾値は train/validation で決める（test で選ばない）。

### その先（prediction-design 参照・合意後）
- **M2.5/M3**: クロスセクション（相対）予測、銘柄間 lead-lag（市場中立化残差）。per-target モデルや銘柄数拡大も。
- **M4**: バックテスト＋根拠説明（金融的評価: 累積リターン・シャープ・最大DD）。
- **M5**: Streamlit ダッシュボード（`[app]` extra に streamlit 済み）。

---

## 6. 未決事項 / 引き継ぎ時の注意

- **閾値 k** は分布を見て調整可（現状 k=0.5 で FLAT 43%）。`--label-mode fixed` でも比較できる。
- **対象銘柄**は大型15銘柄（universe.yaml）。クロスセクション/ lead-lag を狙うなら銘柄数拡大が効く。
- **predictions テーブルの PK=(date,code)** はベースラインと本モデルで上書き衝突しうる。DB 保存を本格化するなら
  `model`/`run` 列の追加を検討（現状 train_baseline は DB 保存せずメトリクス出力のみ）。
- **J-Quants 無料枠の遅延**が評価に与える影響は未確認（現状 yfinance で取得）。
- **class_weight**: 既定 none。balanced で macro-F1↑/accuracy↓のトレードオフ。
- pandas は **3.0 系**（旧 macOS venv は 2.3.3）。取得〜特徴量は動作確認済みだが、新規コードで 3.0 破壊的変更に注意。

---

## 7. リポジトリ / Git 運用

- `main` に直コミット＋ `origin/main` へ push する運用（個人 PoC）。実装ステップ完了ごとに 1 コミット。
- 認証は **HTTPS**（Windows は Git Credential Manager）。Git 識別子はこのリポジトリ限定で
  `YukiOnuma <onumax99@gmail.com>`。
- git 管理外: `.venv/`, `.env`, `data/`（`.gitignore` 済み）。
