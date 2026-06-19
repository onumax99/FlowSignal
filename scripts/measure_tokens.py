"""M3 step ①: count_tokens（無料）で m3-design §6 のトークン仮定を実測し、コストを再計算する。

何をするか:
  1. 固定プレフィックス（指示＋few-shot＋universe 30 銘柄表）と記事可変部の実トークンを測る。
  2. 各モデルの prompt-cache 最小プレフィックスを下回らないか検証する
     （下回ると cache は「無言で効かない」＝0.1× 読込の前提が崩れる）。
  3. 実測値で 1 記事あたり単価・日次 forward / 履歴一括コストを再計算（Batch・cache 込み）。

注意:
  - `count_tokens` は推論しないため**無料**だが、API 呼び出しなので ANTHROPIC_API_KEY は必要。
  - 実ニュース履歴がまだ無いため、ここでは代表的な日本語サンプル記事で概算する。
    実データが入ったら `--articles <path>`（空行区切りの UTF-8 テキスト）で差し替える。
  - 出力トークンは生成しないと測れないため §6.1 の仮定（既定 200）を使う。`--output-tokens` で変更可。

使い方（PowerShell）:
    $env:ANTHROPIC_API_KEY = "sk-ant-..."   # .env でも可
    $env:PYTHONUTF8 = "1"
    python scripts/measure_tokens.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flowsignal import config

# Claude API スキルの正準値（per 1M tokens, 2026-06）。
PRICING = {
    "claude-opus-4-8": {"in": 5.0, "out": 25.0},
    "claude-sonnet-4-6": {"in": 3.0, "out": 15.0},
    "claude-haiku-4-5": {"in": 1.0, "out": 5.0},
}
# prompt-cache の最小キャッシュ可能プレフィックス（これ未満は無言で cache されない）。
CACHE_MIN_TOKENS = {
    "claude-opus-4-8": 4096,
    "claude-sonnet-4-6": 2048,
    "claude-haiku-4-5": 4096,
}
CACHE_READ_MULT = 0.1   # キャッシュ読込 ≈ 0.1×
CACHE_WRITE_MULT = 1.25  # キャッシュ書込 ≈ 1.25×（5分 TTL）
BATCH_MULT = 0.5        # Batch API は全トークン 50% 引

# §6 の仮置き（実取得数で上下する）。
ARTICLES_PER_DAY = 100
HISTORY_ARTICLES = 6000

# few-shot は「方向を当てない」例だけにする（m3-design §3）。
FEW_SHOT = [
    (
        "好決算で通期見通しを上方修正、増配も発表",
        '{"sentiment": 0.8, "impact": 0.9, "novelty": 0.7, "event_type": "earnings", '
        '"rationale": "上方修正＋増配は強いポジ材料。ただし翌日株価の方向は予測しない。"}',
    ),
    (
        "新型モデルを公開、量産は来期以降の見込み",
        '{"sentiment": 0.3, "impact": 0.4, "novelty": 0.5, "event_type": "product", '
        '"rationale": "話題性はあるが業績寄与は限定的・先の話。方向は当てない。"}',
    ),
]

# 実データが無い間の代表サンプル（タイトル＋要約）。日本語の概算用。
SAMPLE_ARTICLES = [
    "トヨタ自動車、第3四半期決算を発表 営業利益は市場予想を上回る。"
    "北米販売が堅調で通期見通しを据え置いた。為替の前提は1ドル150円。",
    "ソフトバンクグループ、保有株の一部売却を発表。"
    "得られた資金は自社株買いと有利子負債の圧縮に充てる方針と説明した。",
    "半導体関連株が軒並み上昇。米ハイテク株高と円安進行を受けて"
    "東京エレクトロンやアドバンテストなど主力銘柄に買いが波及した。",
    "日本銀行は金融政策決定会合で現状維持を決定。"
    "市場の一部にあった追加利上げ観測は後退し、円相場は小幅に下落した。",
]


def build_system_prefix(universe) -> str:
    """指示＋出力スキーマ＋few-shot＋universe 銘柄表からなる固定プレフィックスの草案を組み立てる。

    最終プロンプト（step ④ の prompts/）ではなく、§6 のトークン実測用の代表草案。
    銘柄表は実 universe.yaml から生成し、実データに即した規模を測る。
    """
    lines: list[str] = []
    lines.append(
        "あなたは日本株の材料記事を採点するアナリストです。"
        "記事テキストのみで判断し、後知恵や外部知識を使わないでください。"
        "翌日の株価が上がるか下がるか（方向）は予測しないでください。"
        "出力は sentiment[-1,1] / impact[0,1] / novelty[0,1] / event_type / "
        "relevant_codes（下表の証券コード）/ rationale（120字以内）に限定します。"
    )
    lines.append("")
    lines.append("# few-shot（いずれも方向は当てない）")
    for title, out in FEW_SHOT:
        lines.append(f"記事: {title}")
        lines.append(f"出力: {out}")
    lines.append("")
    lines.append("# universe（証券コード: 銘柄名）")
    for s in universe.stocks:
        lines.append(f"{s.code}: {s.name}")
    return "\n".join(lines)


def _count(client, model: str, content: str, system: str | None = None) -> int:
    kwargs = {"model": model, "messages": [{"role": "user", "content": content}]}
    if system is not None:
        kwargs["system"] = system
    return client.messages.count_tokens(**kwargs).input_tokens


def _per_article_input_cost(prefix_tok: int, article_tok: int, model: str, cacheable: bool) -> float:
    """1 記事あたりの**入力**コスト（$）。cacheable なら steady-state の 0.1× 読込で見積もる。"""
    rate = PRICING[model]["in"] / 1e6
    prefix_mult = CACHE_READ_MULT if cacheable else 1.0
    return (prefix_tok * prefix_mult + article_tok) * rate


def _output_cost(output_tok: int, model: str) -> float:
    return output_tok * PRICING[model]["out"] / 1e6


def load_articles(path: str | None) -> list[str]:
    if not path:
        return SAMPLE_ARTICLES
    text = Path(path).read_text(encoding="utf-8")
    arts = [a.strip() for a in text.split("\n\n") if a.strip()]
    return arts or SAMPLE_ARTICLES


def main() -> None:
    parser = argparse.ArgumentParser(description="count_tokens で §6 のトークン/コストを実測")
    parser.add_argument("--articles", help="空行区切りの UTF-8 テキスト（無ければ内蔵サンプル）")
    parser.add_argument("--output-tokens", type=int, default=200, help="1 記事の想定出力トークン（§6.1）")
    parser.add_argument(
        "--models", nargs="*", default=list(PRICING), help="対象モデル（既定: 3 モデル）"
    )
    args = parser.parse_args()

    if config.anthropic_api_key() is None:
        raise SystemExit(
            "ANTHROPIC_API_KEY が未設定です。.env か環境変数に設定してください"
            "（count_tokens は無料ですが API 呼び出しに鍵が要ります）。"
        )

    import anthropic  # 遅延 import（llm extra）

    client = anthropic.Anthropic()
    universe = config.load_universe()
    prefix = build_system_prefix(universe)
    articles = load_articles(args.articles)

    print(f"universe 銘柄数: {len(universe.stocks)} / サンプル記事数: {len(articles)}")
    print(f"想定出力トークン: {args.output_tokens}（§6.1 の仮定）\n")

    for model in args.models:
        if model not in PRICING:
            print(f"[skip] 未知モデル: {model}")
            continue
        prefix_tok = _count(client, model, prefix)  # system を使わず本文として測る＝プレフィックス規模
        art_toks = [_count(client, model, a) for a in articles]
        mean_art = sum(art_toks) / len(art_toks)

        cache_min = CACHE_MIN_TOKENS[model]
        cacheable = prefix_tok >= cache_min

        in_cost = _per_article_input_cost(prefix_tok, mean_art, model, cacheable)
        out_cost = _output_cost(args.output_tokens, model)
        per_article = in_cost + out_cost

        print(f"=== {model} ===")
        print(f"  固定プレフィックス: {prefix_tok} tok（§6.1 仮定 ~1,500）")
        print(f"  記事可変部(平均):   {mean_art:.0f} tok（§6.1 仮定 ~300, 範囲 {min(art_toks)}–{max(art_toks)}）")
        if cacheable:
            print(f"  cache: ✅ {prefix_tok} ≥ 最小 {cache_min} → 0.1× 読込が有効")
        else:
            print(
                f"  cache: ⚠️ {prefix_tok} < 最小 {cache_min} → **キャッシュされない**"
                f"（プレフィックスを毎回フル課金。few-shot/銘柄表を増やすか cache 前提を外す）"
            )
        print(f"  1 記事単価: ${per_article:.5f}（入力 ${in_cost:.5f} + 出力 ${out_cost:.5f}）")
        print(
            f"  日次 forward(≈{ARTICLES_PER_DAY}記事): "
            f"${per_article * ARTICLES_PER_DAY:.2f}/日"
            f"（+Batch ${per_article * ARTICLES_PER_DAY * BATCH_MULT:.2f}/日）"
        )
        print(
            f"  履歴一括(≈{HISTORY_ARTICLES}件): "
            f"${per_article * HISTORY_ARTICLES:.1f}"
            f"（+Batch ${per_article * HISTORY_ARTICLES * BATCH_MULT:.1f}）"
        )
        print()

    print(
        "→ 実測値で m3-design §6.1/§6.2 を置換し、予算上限を §5.1 に事前登録すること。\n"
        "  キャッシュ非対象（⚠️）の場合は §6 のコストが上振れするので注記を更新する。"
    )


if __name__ == "__main__":
    main()
