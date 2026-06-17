"""評価パッケージ。

- split.py     : 時系列分割（日付境界での walk-forward）
- baselines.py : always-up / random / prev-direction の 3 ベースライン
- metrics.py   : 分類指標（accuracy・macro-F1・per-class・混同行列）＋ McNemar 有意性
"""
