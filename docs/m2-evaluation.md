# FlowSignal M2 評価レポート（テクニカルのみベースライン）

> テクニカル＋マーケット＋カレンダー特徴量のみで翌営業日 3 クラス方向を予測し、
> 3 ベースラインと比較した結果。**M2 の完了条件（ベースライン比較＋有意性）を満たす**。
> 関連: [STATUS.md](STATUS.md) §6 / [prediction-design.md](prediction-design.md) / [requirements.md](requirements.md) §9

- 作成: 2026-06-17
- 再現: `python scripts/train_baseline.py`（balanced 版は `--class-weight balanced`）

---

## 1. セットアップ

| 項目 | 内容 |
|---|---|
| データ | 15銘柄 × 約5年（2021-07〜2026-06）/ 学習テーブル 17,985 行 |
| 特徴量 | 24 列（テクニカル13＋マーケット7＋カレンダー4）。米系/FX は **t-1 overnight** に整合 |
| ラベル | 翌日 close-to-close を 3 クラス。閾値 = **0.5×HV20（ボラ連動）**。分布 DOWN27.2 / FLAT43.0 / UP29.8% |
| モデル | LightGBM 4.6.0（seed=42 固定）。NaN はネイティブ処理 |
| 検証 | **日付境界の walk-forward 5-fold**（同一日付の別銘柄が train/test をまたがない）/ pooled OOS n=15,000 |

リーク防止は各特徴量モジュールの単体テスト（後方参照のみ・先読み無し）と日付境界分割で担保。
→ 下記の数字は **楽観バイアスを排した out-of-sample** の値。

## 2. 結果

| 設定 | accuracy | macro-F1 | fold acc (mean±std) |
|---|---|---|---|
| 既定（class_weight=none） | **41.2%** | 35.1% | 41.2 ± 2.9% |
| balanced（class_weight=balanced） | 39.0% | **36.9%** | 39.0 ± 2.2% |
| ベースライン: prev-direction | 36.3% | – | – |
| ベースライン: random | 33.1% | – | – |
| ベースライン: always-up | 29.6% | – | – |

> ⚠️ **多数派（always-FLAT）ベースラインが未掲載**。FLAT は学習テーブル全体で 43.0%（OOS でも概ね同水準）であり、
> **「常に FLAT」だけで accuracy ≈ 43% に達する＝モデルの 41.2% はこれを上回らない見込み**。下表の有意性は
> あくまで**弱いベースライン（always-up / random / prev-direction）に対する accuracy** の話である。
> また**ベースラインの macro-F1 が未算出**（「–」）のため、方向 skill の実指標で勝っているかは本レポートでは未確認。
> → 追加対応は「[限界・追加検証](#限界追加検証要対応未実施)」を参照。

### McNemar 検定（accuracy の有意差, pooled OOS）

| 比較 | 既定 | balanced |
|---|---|---|
| モデル vs prev-direction | stat=80.9, **p=2.3e-19** | stat=25.0, **p=5.8e-07** |
| モデル vs always-up | stat=352, **p=1.5e-78** | stat=269, **p=2.0e-60** |

→ **3 ベースラインすべてに対し、accuracy で統計的に有意に上回る**（p ≪ 0.001）。

### per-class（pooled OOS, recall）

| クラス | 既定 recall | balanced recall |
|---|---|---|
| DOWN | 0.18 | 0.28 |
| FLAT | 0.67 | 0.50 |
| UP | 0.24 | 0.33 |

## 3. 解釈（正直な評価）

- **有意に勝つが edge は小さい。** 既定設定は accuracy で全ベースラインを有意に上回るものの、
  その多くは **FLAT（多数派 43%）に寄せること**で得た上積み。UP/DOWN の recall は低く、
  **方向当ての真の skill は弱い**。これは設計文書（[prediction-design.md](prediction-design.md) §3、
  [requirements.md](requirements.md) 期待値の前提）の「日次・大型株はほぼ効率的」という想定と整合。
- **accuracy だけ見ると過大評価になる。** balanced にすると accuracy は下がるが macro-F1 と
  UP/DOWN recall が上がる。**macro-F1（35〜37%）が方向 skill の実力**で、改善余地が大きい。
- **効いているのはテクニカルよりマーケット状態。** 重要度上位は vix_level・ret_usdjpy・
  ret_topix・ret_nikkei・chg_vix・米指数リターンが席巻し、自前テクニカル（RSI/MACD/SMA乖離）は下位。
  → テクニカルの作り込みは収穫逓減という想定を裏づけ。

## 限界・追加検証（要対応・未実施）

本レポートの「accuracy で 3 ベースラインに有意」という結論には、以下の限界がある。**いずれも M2.5 で対応する**
（数値は再実行が必要なため本レポートには未反映）。

1. **多数派（always-FLAT）ベースラインの欠落**: 3 クラス不均衡で最も標準的な trivial baseline が比較表に無い。
   FLAT 43.0% に対しモデル accuracy 41.2% なので、**モデルは accuracy では多数派予測器を上回らない見込み**。
   「有意に勝つ」は弱ベースライン限定の主張である点を明示する。
2. **ベースラインの macro-F1 未算出（指標で強弱が逆転）**: 方向 skill の実指標（macro-F1）でベースラインと比較していない。
   概算では **always-FLAT の macro-F1 ≈0.20（accuracy では最強だが macro-F1 では最弱の strawman）/ random-uniform ≈0.33 /
   prev-direction ≈0.33〜0.36（macro-F1 の本当の対戦相手）**。モデル既定 0.351 は **prev-direction に負け得る**。
   accuracy 側で弱ベースラインに勝ったのと対称に、macro-F1 で「多数派に勝った」と言っても中身は薄い。**全ベースラインの macro-F1 を併記**し、指標ごとに対称評価する。
3. **macro-F1 の有意性検定が無い**: McNemar は accuracy 専用。本命指標である macro-F1 については、
   **日次（または連続週）ブロックの block bootstrap で信頼区間**を出し、ベースラインを超えるかをリークなく検証する。
   再標本化の単位は**日（同一日の銘柄間相関を保つ 1 ブロック）であって 5 fold ではない**（fold を resample すると単位が 5 個で CI が不安定）。

→ 追加対応: ① `eval/baselines.py` に `always-majority(FLAT)` を追加 ② 全ベースラインの macro-F1 を表に併記
③ macro-F1 の bootstrap CI を算出。これらを M2.5 で実施し、本レポートを更新する。

## 4. M2 の結論と M3 への示唆

- **M2 完了条件は達成**: テクニカルのみで方向予測し、3 ベースラインと比較・有意性確認まで実施。
  パイプライン（取得→特徴量→ラベル→学習→walk-forward 評価）はリーク管理込みで通る。
- **精度を上げる次の一手はモデルいじりではない**（[prediction-design.md](prediction-design.md) §4）:
  1. **HOLD（確信度で棄権）＋確率較正** — 高確信日だけ予測し実用精度を上げる（低コスト・最優先）
  2. **クロスセクション（相対）予測** — 市場共通要因を外し銘柄固有シグナルを学習しやすくする
  3. **イベント駆動の LLM 新情報（M3 本命）** — 決算・開示・ニュースで価格に未織り込みの edge を取る
- 現状の重要度（マーケット系が支配的）は、上記 2・3 の方向性（相対化・新情報）の妥当性を支持する。

> 注: McNemar は accuracy（正誤の 2×2）に対する検定で macro-F1 は対象外。fold 間 mean±std は
> fold が相関するため厳密な検定ではない（PoC のヒューリスティック）。
