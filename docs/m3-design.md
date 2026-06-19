# FlowSignal M3 設計書（LLM ニュース/開示特徴量・初の Claude API 課金）

> M3 の詳細設計とコスト見積もり。**実 API 実行（課金）の手前まで**を確定する。
> 関連: 要件 [requirements.md](requirements.md) §8.2 / 引き継ぎ [handoff.md](handoff.md) §5 / 検討メモ [prediction-design.md](prediction-design.md) §3・§4③ / [m2-evaluation.md](m2-evaluation.md)
> - 作成: 2026-06-19 / ステータス: **設計確定待ち**（合意後に実装へ）

---

## 0. なぜ M3 が本命か（30 秒）

M2.5 で **価格・テクニカルの後処理（HOLD・確率較正・クロスセクション）はいずれも negative**（macro-F1/IC を動かせず）と確定した。
方向 skill に効いた唯一のレバーは raw balanced の macro-F1（弱いが本物）だけ。残るレバーは **新情報＝ニュース/適時開示** で、
ここで初めて Claude API（有料）を使う。**M3 はこの PoC の正念場**であり、無料工程をやり切った今こそ課金の根拠が明確。

LLM の役割は**推論専用**（ファインチューニングしない）。Claude にニュース/開示を採点させて**特徴量を出力**させ、
それを既存の LightGBM の入力に足す。学習するのは引き続き LightGBM のみ（要件 §8.2）。

---

## 1. データ方針の決定（最重要・リーク二軸）

M3 のデータには **(A) 可用性** と **(B) LLM hindsight** の二軸があり、両者は独立に効く。

| | (A) データ可用性 | (B) LLM hindsight |
|---|---|---|
| RSS（現状 `data/news.py`） | ❌ 直近のみ・過去アーカイブ無し | — |
| TDnet 適時開示（履歴取得可） | ✅ 過去取得可 | ❌ `claude-opus-4-8` のカットオフ 2026-01 以前を採点すると結末を「知っている」未来情報が混入 |
| forward-only（今から収集） | △ 小標本（日々増える） | ✅ カットオフ後＝モデル未学習＝構造的にリーク無し |

**決定**: 
- **評価（合否判定）は forward-only を本筋**にする。カットオフ後の記事/開示だけが hindsight クリーン。
- **TDnet 履歴は「特徴量パイプラインの開発・デバッグ用」**に使う（スコア化器・結合・テストを過去データで作り込む）。
  ただし**履歴で出した backtest 数値は hindsight 前提の参考値**として明記し、合否には使わない。
- forward 区間が貯まるまでは「**条件付き（材料のある日）**の小標本での暫定評価」に留める。

> これは prediction-design §2.1-5・requirements §13 の「forward-only が唯一のリーク無し評価」を M3 の運用方針として確定するもの。

---

## 2. LLM スコア化の仕様（何を出力させるか）

### 2.1 出力スキーマ（記事 → 構造化スコア）
1 記事につき以下を **structured outputs（`output_config.format` の json_schema）** で固定して取り出す:

| フィールド | 型 | 説明 |
|---|---|---|
| `relevant_codes` | string[] | universe 30 銘柄のうち記事が**直接言及/明確に関係**する証券コード（0〜数件） |
| `sentiment` | number[-1,1] | 記事の論調（ポジ/ネガ）。**方向予測ではなくテキストの論調** |
| `impact` | number[0,1] | 記事の重要度（業績・需給に効きそうか） |
| `novelty` | number[0,1] | 既知情報の繰り返しか新規か（織り込み度の代理） |
| `event_type` | enum | `earnings`/`guidance`/`mna`/`rating`/`product`/`macro`/`other`（開示・ニュース種別） |
| `rationale` | string(≤120) | 採点根拠の短い説明（FR-6 の根拠提示にも流用） |

### 2.2 リーク防止（ニュース特有・必須）
- **時点整合**: 記事の `published` が **JP 大引け t 以前**のものだけを row t に使う（t の終値後の記事は t+1 行へ）。`market.py` と同じ思想で固定＋テスト。
- **LLM hindsight 対策**（プロンプトで明示）:
  1. **記事テキストのみで判断**・後知恵/外部知識を使わない
  2. **方向（上がる/下がる）を予測させない** — 出力は sentiment/impact/novelty/関連度の定量化に限定（方向当ては LightGBM の役割）
  3. **日付を伏せて採点**（プロンプトに日付・"as of" を入れない）
  4. 構造的には **forward-only 区間が本命**（§1）。履歴採点は参考値扱い。

### 2.3 モデル選定（既定は Opus 4.8・コストは §6 で選択）
- 既定 **`claude-opus-4-8`**（$5/$25 per 1M）。スキルの方針どおりコスト目的の独断ダウングレードはしない。
- **モデル階層化の選択肢**（ユーザー判断）: 大量のスコア化を `claude-sonnet-4-6`（$3/$15）や `claude-haiku-4-5`（$1/$5）に、
  少数の根拠生成（rationale を読み物にする場合）を Opus に。§6 のコスト表で比較し合意する。
- 思考は adaptive（`thinking: {type:"adaptive"}`）、構造化抽出なので effort は `low`〜`medium` で十分。

---

## 3. プロンプト設計

- **固定プレフィックス（prompt caching 対象）**: 指示文＋few-shot（3〜5 例）＋ universe 銘柄表（コード/別名）。
  バイト単位で固定し `cache_control:{type:"ephemeral"}` を最後の固定ブロックに置く（読込は約 0.1× 課金）。
  → 日次バッチや一括採点で**指示文を毎回フルで払わない**。
- **可変部（キャッシュ後）**: 記事の title＋summary のみ（日付・URL は入れない＝hindsight 低減）。
- **出力固定**: `output_config.format` の json_schema（§2.1）または `messages.parse()`＋Pydantic。
  refusal（`stop_reason=="refusal"`）時はスキップして欠損特徴量にする（リーク防止上も安全）。
- few-shot は**方向を当てない**例だけにする（「この記事は強いポジ材料だが**翌日株価は予測しない**」を体現）。

---

## 4. 特徴量化・結合（build テーブルへ）

記事スコアを **(date, code) 粒度**へ集約して既存 24 特徴量に時点整合で結合する。

- **時点整合**: `published ≤ 大引け t` の記事のみを (t, code) に割当（§2.2）。
- **日次集約（銘柄ごと）**: `news_impact_max`, `news_sentiment_mean`(impact 重み), `news_novelty_max`,
  `news_count`, `has_event`(当日材料フラグ), `event_type` ワンホット主要種別。
- 欠損（材料なしの日）は 0 / フラグ off（lightgbm は NaN ネイティブだが意味的に 0 が自然な列は 0）。
- 保存: `data/processed/news_scores.parquet`（記事粒度・スコア生値）＋集約は build 時に算出。
  記事粒度を残すことで再集約・デバッグ・根拠提示が可能。

---

## 5. 評価計画（M2 比の改善幅）

- **同じ walk-forward ハーネス**（日付境界・balanced・macro-F1 が本命）で **M2（24 特徴量）vs M2+LLM** を比較。
- **pooled だけでなく「材料のある日」条件付き**でも測る（イベント効果は pooled だと希釈される）。
- **有意性**: macro-F1 の **日付ブロック bootstrap**（既存 `eval/metrics.block_bootstrap_macro_f1` を再利用）で
  Δ(=M2+LLM − M2) の CI を出す。**CI 下限 > 0 で有意**。
- クロスセクション版も任意で（rank IC の差・既存 `eval/cross_section_metrics`）。

### 5.1 成功基準の事前登録（pre-register・着手前に凍結）
> M2.5 が一貫 negative だった以上、後から良い指標を選ばないため**着手前に合否を固定**する。

**M3 成功 = 次を満たすこと**（forward-only または材料のある日条件で）:
1. **macro-F1**: balanced 設定で **Δ(M2+LLM − M2) の日付ブロック bootstrap 95%CI 下限 > 0**（材料のある日サブセット）。
2. 同時に **pooled でも悪化しない**（pooled Δ の CI 下限 ≥ −ε, ε=0.005 程度）。
3. （任意・差別化確認）イベント当日に絞ると効果が増える（条件付き Δ > pooled Δ）。

満たさなければ **negative as M2.5**（価格＋公開ニュースでは edge 無し）として正直に記録し、深追いしない。

---

## 6. コスト見積もり（Batch・caching・階層化込み）

> 価格は Claude API スキルの正準値（per 1M tokens, 2026-06 時点）: **Opus 4.8 $5/$25・Sonnet 4.6 $3/$15・Haiku 4.5 $1/$5**。
> Batch API は**全トークン 50% 引**。prompt caching は**読込 ≈0.1×・書込 1.25×(5分)**。**実数は着手時に `count_tokens` で確定**（下記 §6.3）。

### 6.1 トークン仮定（1 記事あたり・**要 count_tokens 確定**）
- 固定プレフィックス（指示＋few-shot＋銘柄表, キャッシュ）: **~1,500 tok**（毎回 0.1× 読込）
- 可変入力（title＋summary, 日本語）: **~300 tok**（フル課金）
- 出力（構造化 JSON＋短い rationale）: **~200 tok**

### 6.2 概算（1 記事の単価と日次/一括）
1 記事単価 = キャッシュ読込(1,500×0.1) + 入力300 + 出力200 を各レートで:

| モデル | 1 記事 | 日次 forward(≈100 記事/日) | 同 + Batch50% | TDnet 履歴一括(≈6,000 件) | 同 + Batch50% |
|---|---|---|---|---|---|
| Opus 4.8 | ~$0.0073 | **~$0.73/日** | ~$0.36/日 | ~$44 | ~$22 |
| Sonnet 4.6 | ~$0.0044 | ~$0.44/日 | ~$0.22/日 | ~$26 | ~$13 |
| Haiku 4.5 | ~$0.0015 | ~$0.15/日 | ~$0.07/日 | ~$9 | ~$4.5 |

- **forward 運用**は Opus でも概ね **$0.4〜0.7/日**（requirements §8.2 の「$0.5〜1.5/日」と整合）。
- **履歴一括**（TDnet 5 年・開発用）は **一回 $5〜44**（モデルと Batch で変動）。forward-only 本筋なら一括採点は不要〜小。
- 記事数（100/日・6,000 件）は仮置き。RSS 実取得数・TDnet 開示数で上下する。

### 6.3 確定方法（無料・着手時）
- `count_tokens` は**無料**（推論しない）。実記事サンプルで:
  ```python
  from anthropic import Anthropic
  client = Anthropic()  # ANTHROPIC_API_KEY が必要
  n = client.messages.count_tokens(model="claude-opus-4-8",
        system=SYSTEM_PREFIX, messages=[{"role":"user","content":sample_article}]).input_tokens
  ```
  日本語は英語より多めに出るため、**固定プレフィックスと記事 20〜50 件で実測**して §6.1 を置換 → §6.2 を再計算。
- **予算上限を事前登録**（例: 開発一括 ≤ $X、月次 forward ≤ $Y）。超えそうなら Haiku 階層化や記事フィルタ（impact 下限）で抑制。

---

## 7. 新規モジュール・実装計画（合意後）

1. `config.anthropic_api_key()` … `.env` の `ANTHROPIC_API_KEY` 読み込み（`jquants_credentials` と同型）。`.env.example` に追記。
2. `data/disclosure.py` … **TDnet 適時開示の取り込み**（履歴取得・正規化、news と同じ共通スキーマ）。**M3 必須**・汎用 RSS より優先（差別化の本命）。
3. `features/news_llm.py` … Claude でスコア化（structured outputs＋prompt caching＋Batch 対応）、時点整合の固定＋テスト。
4. `prompts/` … システム指示＋few-shot＋json_schema 定義（バージョン管理）。
5. スコア保存 `data/processed/news_scores.parquet`、build への集約結合（§4）。
6. `scripts/score_news.py`（採点バッチ）/ `scripts/train_with_news.py`（M2+LLM 評価）。
7. テスト: 時点整合（published≤t）・集約・refusal スキップ・スキーマ検証（API はモックして無料でテスト）。

**着手順**: ①config 鍵＋count_tokens で §6 確定・予算登録 → ②disclosure.py（データ実在性の決着）→ ③news_llm.py（少数記事で疎通）→ ④結合・評価ハーネス → ⑤forward 収集開始＆条件付き暫定評価。

---

## 8. リスク・未決

- 🚨 **TDnet 取得の実在性/規約**: 履歴取得の可否・形式・利用規約を着手前に確認（取得できなければ forward-only のみで進める）。
- **forward 標本の貯まり待ち**: forward-only は評価に時間がかかる。まず「材料のある日」条件付きの小標本で兆候を見る。
- **コスト統制**: 鍵の管理（`.env`・git 管理外）、予算上限の事前登録、impact 下限での記事フィルタ。
- **refusal/障害**: `stop_reason=="refusal"` や API エラーはスキップして欠損特徴量化（リーク・コスト両面で安全）。
- **モデル選定はユーザー判断**: 既定 Opus 4.8。コスト目的の階層化（Sonnet/Haiku）は §6 を見て合意してから。
