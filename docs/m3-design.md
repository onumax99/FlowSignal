# FlowSignal M3 設計書（LLM ニュース/開示特徴量・初の Claude API 課金）

> M3 の詳細設計とコスト見積もり。**実 API 実行（課金）の手前まで**を確定する。
> 関連: 要件 [requirements.md](requirements.md) §8.2 / 引き継ぎ [handoff.md](handoff.md) §5 / 検討メモ [prediction-design.md](prediction-design.md) §3・§4③ / [m2-evaluation.md](m2-evaluation.md)
> - 作成: 2026-06-19 ／ 更新: 2026-06-19（pre-register 安全策①〜⑤を反映: ①第3結末 inconclusive・②開封点固定で逐次のぞき見防止・③履歴は配管検証のみ・④検定力の単位を分離・⑤カットオフ buffer） / ステータス: **着手可（M3 step ①〜・安全策反映済み）**

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
  ⑤ ただし**知識カットオフは硬い境界ではなく前後がにじむ**ため、保守的に **カットオフ＋バッファ（目安 ≥4 週間）以降のみ「真にクリーン」**として合否に用いる（直後の月は除外）。バッファ幅は §5.1 で事前登録して凍結する。
- ③ **TDnet 履歴は「特徴量パイプラインの配管検証用」に限定**する（パース・時点整合テスト・refusal 処理・結合の正しさを過去データで確認する用途のみ）。
  🚨 **スコア分布に影響する選択（プロンプト文言・impact 下限・残す特徴量・集約方法）は、履歴の backtest 成績を見る前に凍結する**。履歴成績が良くなる方向に調整すると hindsight がその選択に染み込み forward を汚す（＝ prediction-design §2.1 のスヌーピング規律を M3 に適用）。**履歴で出した backtest 数値は hindsight 前提の参考値**として明記し、**合否にも選択（チューニング）にも使わない**。
- forward 区間が貯まるまでは「**条件付き（材料のある日）**の小標本での暫定評価」に留める（合否は §5.1 の開封点まで判定しない）。

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

- **固定プレフィックス**: 指示文＋few-shot（3〜5 例）＋ universe 銘柄表（コード/別名）。
  🆕 **実測 745 tok（Opus）/ 713（Sonnet・Haiku）＝全モデルの prompt-cache 最小（Opus/Haiku 4,096・Sonnet 2,048 tok）を下回る** → **prompt caching は使わない**（小さなプレフィックスを最小値まで水増しすると逆に高くつくため）。コストの主レバーは **Batch API（50% 引）**にする（§6）。プレフィックスは小さいので毎回フル課金でも 1 記事 ~$0.004（Opus）。
- **可変部**: 記事の title＋summary のみ（日付・URL は入れない＝hindsight 低減）。
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

① **3 値で結論する（negative と「測定不能」を混同しない）**:
- **positive**: 上記 1（＋2）を満たす。
- **negative**: **最小サンプル N_min に到達した上で** 1 を満たさない＝価格＋公開ニュースでも edge 無し（M2.5 と同じ正直な negative）。
- **inconclusive（検定力不足）**: N_min 未到達。**negative とは書かない**。forward 継続かスコープ縮小かを判断する。

② **開封点を 1 点に固定する（逐次のぞき見＝optional stopping の防止）**:
- forward はデータが日々増えるため、**届くたびに再評価して「CI が 0 を割れたら成功」と止めると多重検定で偽陽性が膨らむ**（事前登録が無効化される）。
- → **開封点を「材料のある (date,code) 行数 ≥ N_min」または「評価日 = YYYY-MM-DD」の 1 点に事前固定し、それまで forward の合否数値を見ない**。N_min・評価日・§1 のカットオフバッファ幅は **着手時に確定して凍結**する（plumbing 確認のため途中で見る場合も合否判定はしない）。
- N_min は §6.3 で実測する forward の**材料イベント発生率**から、日付ブロック bootstrap が安定する規模（目安: 材料のある (date,code) 行が数百以上）を逆算して決める。

---

## 6. コスト見積もり（Batch・caching・階層化込み）

> 価格は Claude API スキルの正準値（per 1M tokens, 2026-06 時点）: **Opus 4.8 $5/$25・Sonnet 4.6 $3/$15・Haiku 4.5 $1/$5**。
> Batch API は**全トークン 50% 引**（M3 のコスト主レバー）。prompt caching は**プレフィックスが各モデルの最小を下回るため不使用**（§3）。**実数は count_tokens で実測済み（2026-06-19・`scripts/measure_tokens.py`）**。

### 6.1 トークン仮定（**count_tokens 実測済み 2026-06-19**・`scripts/measure_tokens.py`）
- 固定プレフィックス（指示＋few-shot＋30 銘柄表）: **実測 745 tok（Opus 4.8）/ 713（Sonnet 4.6・Haiku 4.5）**（仮定 ~1,500 から減）。**cache 最小未満につき毎回フル課金**（§3）。
- 可変入力（title＋summary, 日本語）: **実測 ~67 tok**（短いサンプル 4 件・範囲 62–74）。⚠️ **実 RSS/TDnet 要約はより長い可能性**＝step ② で実記事を `--articles` に渡して再実測する。
- 出力（構造化 JSON＋短い rationale）: **~200 tok（仮定・生成しないと測れない）**。

### 6.2 概算（**実測値・cache 不使用 / Batch あり**）
1 記事単価 = プレフィックス(745 or 713)×1.0 + 入力67 + 出力200 を各レートで:

| モデル | 1 記事 | 日次 forward(≈100 記事/日) | 同 + Batch50% | TDnet 履歴一括(≈6,000 件) | 同 + Batch50% |
|---|---|---|---|---|---|
| Opus 4.8 | ~$0.0091 | ~$0.91/日 | **~$0.45/日** | ~$54 | ~$27 |
| Sonnet 4.6 | ~$0.0053 | ~$0.53/日 | ~$0.27/日 | ~$32 | ~$16 |
| Haiku 4.5 | ~$0.0018 | ~$0.18/日 | ~$0.09/日 | ~$11 | ~$5.3 |

- 🚨 **$5 予算では「TDnet 履歴一括(6,000 件)」は全モデルで赤字**（Haiku+Batch でも ~$5.3）。→ **履歴一括採点はしない**。履歴は配管検証用に**数十件サンプルのみ採点（数セント）**に留める（§1「履歴は配管検証のみ」と整合）。
- **forward 運用**は 100 記事/日・Batch で **Opus $0.45 / Sonnet $0.27 / Haiku $0.09 per 日**。$5 で概ね **Opus ~11 日 / Sonnet ~18 日 / Haiku ~55 日**ぶん。
- ④ 記事数（100/日）は仮置きで、実際に universe 30 銘柄へ紐づく行数はずっと少ない（§6.3）。可変入力 67 tok も短サンプル由来＝**実記事で上振れしうる**。**forward の実単価は step ② の実データで再確定**。

### 6.3 確定方法と予算（count_tokens は実測済み）
- ✅ **count_tokens 実測済み（2026-06-19・`scripts/measure_tokens.py`）**: §6.1/§6.2 を実測値に置換済み。`count_tokens` は無料（推論しない）。実記事が入ったら `--articles` で再実測する。
- 🚨 **予算上限を事前登録 ＝ $5 ハードキャップ**（追加クレジット $5）。方針:
  **(a) 履歴一括採点はしない**（§6.2 で赤字）／**(b) forward-only ＋ Batch（50%）**／
  **(c) 配管検証は履歴を数十件サンプルのみ採点（数セント）**／**(d) 記事フィルタ（impact 下限）で forward 件数を抑制**。
  モデル選定（Opus/Sonnet/Haiku）は runway に直結（§6.2: $5 で Opus ~11 日 / Sonnet ~18 日 / Haiku ~55 日）＝**ユーザー判断**。
- ④ **材料イベント発生率の実測（検定力＝開封点 N_min の根拠）**: ⏳ **未実施（実ニュース/開示データ待ち＝step ②）**。データ取得後に「**非ゼロ news 特徴量を持つ (date,code) 行 / 全行**」を数え、forward で N_min（数百行）到達の所要日数を見積もる。§5.1 の開封点・評価日はこの実測で確定して凍結する。

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

- ✅ **TDnet 取得の実在性/規約（2026-06-19 決着）**: 履歴(5年)は**有料のみ**（公式 TDnet API／J-Quants アドオン ¥11,000 月・Light 以上・個人投資家限定）、**forward は release.tdnet.info の無料 31 日窓で取得可**。→ **forward-only を無料経路で実装**、履歴一括はしない（§6 で $5 赤字）。配管検証も無料 31 日窓で足りる。`data/disclosure.py` のソース＝release.tdnet.info（規約・構造変化に注意）。
- **forward 標本の貯まり待ち**: forward-only は評価に時間がかかる。まず「材料のある日」条件付きの小標本で兆候を見る。
- **コスト統制**: 鍵の管理（`.env`・git 管理外）、予算上限の事前登録、impact 下限での記事フィルタ。
- **refusal/障害**: `stop_reason=="refusal"` や API エラーはスキップして欠損特徴量化（リーク・コスト両面で安全）。
- **モデル選定はユーザー判断**: 既定 Opus 4.8。コスト目的の階層化（Sonnet/Haiku）は §6 を見て合意してから。
