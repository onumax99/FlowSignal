# FlowSignal M2 評価レポート（テクニカルのみベースライン）

> テクニカル＋マーケット＋カレンダー特徴量のみで翌営業日 3 クラス方向を予測し、
> **4 ベースライン（多数派＋弱 3 種）**と accuracy / macro-F1 の両指標で比較した結果。
> **M2 の完了条件を満たし、M2.5 の honest 化（多数派・全 macro-F1・bootstrap）まで反映済み**。
> 関連: [STATUS.md](STATUS.md) §6 / [prediction-design.md](prediction-design.md) / [requirements.md](requirements.md) §9

- 作成: 2026-06-17 ／ 更新: 2026-06-18（**adjusted close 再実行＋ honest 化**＝多数派ベースライン・全 macro-F1・日付ブロック bootstrap を追加）
- 再現: `python scripts/train_baseline.py`（balanced 版は `--class-weight balanced`、bootstrap 反復は `--n-boot`）

---

## 1. セットアップ

| 項目 | 内容 |
|---|---|
| データ | 15銘柄 × 約5年（2021-07〜2026-06）/ 学習テーブル 17,985 行。株価は **adjusted close（yfinance `auto_adjust=True`）**＝分割・配当調整後 |
| 特徴量 | 24 列（テクニカル13＋マーケット7＋カレンダー4）。米系/FX は **t-1 overnight** に整合 |
| ラベル | 翌日 close-to-close を 3 クラス。閾値 = **0.5×HV20（ボラ連動）**。分布 DOWN27.0 / FLAT43.0 / UP30.0% |
| モデル | LightGBM 4.6.0（seed=42 固定）。NaN はネイティブ処理 |
| 検証 | **日付境界の walk-forward 5-fold**（同一日付の別銘柄が train/test をまたがない）/ pooled OOS n=15,000 |

リーク防止は各特徴量モジュールの単体テスト（後方参照のみ・先読み無し）と日付境界分割で担保。
→ 下記の数字は **楽観バイアスを排した out-of-sample** の値。

## 2. 結果

| 設定 | accuracy | macro-F1 | fold acc (mean±std) |
|---|---|---|---|
| 既定（class_weight=none） | 41.0% | 0.349 | 41.0 ± 2.4% |
| balanced（class_weight=balanced） | 39.6% | **0.372** | 39.6 ± 2.6% |
| ベースライン: always-majority(FLAT) | **43.4%** | 0.202 | – |
| ベースライン: prev-direction | 36.3% | 0.345 | – |
| ベースライン: random | 33.0% | 0.326 | – |
| ベースライン: always-up | 29.8% | 0.153 | – |

> **指標で勝者が入れ替わる（honest 比較・M2.5 で多数派＋全 macro-F1 を追加）**:
> - **accuracy では多数派(always-FLAT 43.4%)が最強**。モデルは既定 41.0%・balanced 39.6% で
>   **多数派に負ける**（McNemar でも多数派が有意に上）。M2 当初の「accuracy で有意」は**弱ベースライン限定**だった。
> - **macro-F1（方向 skill の本命指標）では多数派は 0.202 と最弱**。モデルは全ベースラインを上回るが、
>   既定 0.349 は最強の macro-F1 baseline である prev-direction(0.345)と**統計的に並ぶ（差は有意でない）**。
>   **balanced 0.372 は prev-direction 含む全ベースラインに有意**（下記 bootstrap）。

### McNemar 検定（accuracy の有意差, pooled OOS）

| 比較 | 既定 | balanced |
|---|---|---|
| モデル vs always-majority | stat=35.0, **p=3.2e-09**（多数派優位） | stat=58.7, **p=1.8e-14**（多数派優位） |
| モデル vs prev-direction | stat=74.0, **p=7.9e-18**（モデル優位） | stat=35.3, **p=2.9e-09**（モデル優位） |
| モデル vs always-up | stat=331, **p=4.6e-74**（モデル優位） | stat=286, **p=3.5e-64**（モデル優位） |

→ accuracy では **多数派に有意に負け**、弱 2 種には有意に勝つ。**accuracy は不均衡 3 クラスでは方向 skill を測れない**（多数派が最強）。

### macro-F1 の有意性（日付ブロック bootstrap, n_boot=2000）

McNemar は accuracy 専用なので、本命指標 macro-F1 は**日付ブロック bootstrap**（5 fold ではなく**日**を再標本化）で
95%CI と model−baseline の差 Δ を出す。**Δ の CI 下限 > 0 で有意**。

| 系列 | macro-F1 [95%CI] | Δ vs 既定 | Δ vs balanced |
|---|---|---|---|
| model（既定） | 0.349 [0.337, 0.360] | – | – |
| model（balanced） | 0.372 [0.361, 0.383] | – | – |
| prev-direction | 0.345 [0.335, 0.356] | +0.004 [−0.012, +0.019] **有意差なし** | +0.027 [+0.010, +0.042] **有意** |
| random | 0.326 [0.319, 0.334] | +0.022 [+0.008, +0.035] 有意 | +0.045 [+0.032, +0.059] 有意 |
| always-majority | 0.202 [0.198, 0.206] | +0.147 有意 | +0.170 有意 |
| always-up | 0.153 [0.147, 0.159] | +0.196 有意 | +0.219 有意 |

→ **balanced は全ベースライン（最強の prev-direction 含む）に macro-F1 で有意**。既定は prev-direction と並ぶ。
**方向 skill は「弱いが本物」**＝最強 baseline 比 Δ≈+0.027（balanced）。小さいが CI が 0 を外す。

### per-class（pooled OOS, recall）

| クラス | 既定 recall | balanced recall |
|---|---|---|
| DOWN | 0.17 | 0.27 |
| FLAT | 0.67 | 0.52 |
| UP | 0.25 | 0.32 |

## 3. 解釈（正直な評価）

- **accuracy では多数派に負ける。** モデル accuracy（既定 41.0% / balanced 39.6%）は always-FLAT(43.4%)を
  下回り、McNemar でも多数派が有意に上。**accuracy は不均衡 3 クラスでは方向 skill を測れない**（多数派が最強）。
  M2 当初の「accuracy で有意」は弱ベースライン限定の主張だった。これは設計文書（[prediction-design.md](prediction-design.md) §3、
  [requirements.md](requirements.md) 期待値の前提）の「日次・大型株はほぼ効率的」という想定と整合。
- **macro-F1 で見ると skill は「弱いが本物」。** 本命指標 macro-F1 では多数派は 0.202 と最弱。
  **balanced(0.372)は最強 baseline の prev-direction(0.345)を含む全ベースラインに日付ブロック bootstrap で有意**
  （Δ≈+0.027, CI[+0.010,+0.042]＝0 を外す）。既定(0.349)は prev-direction と統計的に並ぶ。edge は小さいが、リークを排した上で残る本物の差。
- **balanced を本命設定とする。** accuracy は下がるが macro-F1・UP/DOWN recall が上がり、honest 指標で全ベースライン超え。改善余地は大きい。
- **効いているのはテクニカルよりマーケット状態。** 重要度上位は vix_level・ret_usdjpy・
  ret_topix・ret_nikkei・chg_vix・米指数リターンが席巻。自前テクニカルは hv20/出来高系がかろうじて続く程度で、
  RSI/MACD/SMA乖離は下位。→ テクニカルの作り込みは収穫逓減という想定を裏づけ。

## 限界・追加検証

当初レポートの限界（#1 多数派ベースライン欠落 / #2 全 macro-F1 未算出 / #3 macro-F1 の有意性検定なし /
#4 分割・配当未調整）は **すべて解決済み（2026-06-18・上表に反映）**:

1. ✅ **多数派ベースライン**: `eval/baselines.py` に `always-majority(FLAT)` を追加。多数派は**各 fold の train 最頻クラス**
   （test を見ないリーク無し）。accuracy 43.4% でモデルを上回り、「accuracy で有意」は弱ベースライン限定だったと確認。
2. ✅ **全ベースラインの macro-F1 併記**: always-majority 0.202 / prev 0.345 / random 0.326 / always-up 0.153。
   accuracy と逆に多数派が最弱・prev-direction が最強、と指標で強弱が入れ替わる。
3. ✅ **macro-F1 の有意性検定**: `eval/metrics.block_bootstrap_macro_f1`（**日付ブロック** bootstrap・5 fold ではない）で
   CI と差を算出。balanced は全ベースラインに有意、既定は prev-direction と並ぶ（上表）。
4. ✅ **株価の分割・配当調整**: `data/prices.py` を yfinance `auto_adjust=True`（adjusted close）に切替＋再取得・再実行。
   集計指標の変化は小（既定 accuracy 41.2→41.0%・macro-F1 35.1→34.9%、balanced 39.0→39.6%・36.9→37.2%）で結論は不変だがデータは健全化。

→ 残課題（M3 以降）: J-Quants 経路の adjusted 化（`prices.py` TODO）、確率較正＋HOLD のカバレッジ評価、クロスセクション化（要 universe 拡大）。

## 4. M2 の結論と M3 への示唆

- **M2 完了条件は達成＋ M2.5 honest 化まで実施**: テクニカルのみで方向予測し、4 ベースライン（多数派含む）と
  accuracy / macro-F1 の両指標で比較・有意性確認（McNemar＋日付ブロック bootstrap）まで完了。
  パイプライン（取得→特徴量→ラベル→学習→walk-forward 評価）はリーク管理込みで通る。
- **honest な結論**: accuracy では多数派に負ける（accuracy は方向 skill を測れない）。macro-F1 では
  **balanced が全ベースライン（最強 prev-direction 含む）に有意**＝**方向 skill は弱いが本物**（最強比 Δ≈+0.027）。
- **精度を上げる次の一手はモデルいじりではない**（[prediction-design.md](prediction-design.md) §4）:
  1. **HOLD（確信度で棄権）＋確率較正** — 高確信日だけ予測し実用精度を上げる（低コスト・最優先）
  2. **クロスセクション（相対）予測** — 市場共通要因を外し銘柄固有シグナルを学習しやすくする
  3. **イベント駆動の LLM 新情報（M3 本命）** — 決算・開示・ニュースで価格に未織り込みの edge を取る
- 現状の重要度（マーケット系が支配的）は、上記 2・3 の方向性（相対化・新情報）の妥当性を支持する。

> 注: McNemar は accuracy（正誤の 2×2）に対する検定で macro-F1 は対象外。fold 間 mean±std は
> fold が相関するため厳密な検定ではない（PoC のヒューリスティック）。
