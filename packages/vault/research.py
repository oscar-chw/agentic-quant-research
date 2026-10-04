#!/usr/bin/env python3
"""
research.py — Research CLI for the quant vault.

Commands:
    python research.py "momentum factor crypto"           semantic search
    python research.py --alpha-ideas "pairs trading"      search + Claude alpha generation
    python research.py --stats                            vault statistics
    python research.py --related 2401.12345               find related papers
    python research.py --daily                            today's new papers
    python research.py --top-crypto 10                    top N papers by Crypto score
    python research.py --export "momentum" out.md         export results to markdown
    python research.py --distill                             distill all 12 topics to guidelines/
"""

import argparse
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import chromadb
import yaml

from source_access import current_note, query_note_sources

# ── UTF-8 stdout fix ──────────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.yaml"


# ── Config ────────────────────────────────────────────────────────────────────


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["vault_path"] = str(Path(cfg["vault_path"]).expanduser())
    return cfg


def get_collection(cfg: dict) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=cfg["chroma_path"])
    return client.get_or_create_collection(
        name="quant_papers",
        metadata={"hnsw:space": "cosine"},
    )


def get_db(cfg: dict) -> sqlite3.Connection:
    return sqlite3.connect(cfg["db_path"])


# ── Display helpers ───────────────────────────────────────────────────────────

DIVIDER = "\u2501" * 60  # ━━━...


def print_paper(
    rank_or_score: str,
    title: str,
    arxiv_id: str,
    published: str,
    categories: str,
    excerpt: str,
) -> None:
    print(f"\n\u2501\u2501\u2501 [{rank_or_score}] {title[:70]} \u2501\u2501\u2501")
    date_str = published[:7] if published else "unknown"
    print(f"arXiv: {arxiv_id} | {date_str} | {categories}")
    if excerpt:
        # Clean excerpt: collapse whitespace, limit to 300 chars
        clean = " ".join(excerpt.split())
        print(clean[:300] + ("..." if len(clean) > 300 else ""))


# ── Search ────────────────────────────────────────────────────────────────────


def do_search(cfg: dict, query: str, n: int = 8) -> list[dict]:
    result = query_note_sources(get_collection(cfg), query, n, source_root=cfg["vault_path"])
    print(f"{result['status']}: {len(result['matches'])} eligible local notes; "
          f"{len(result['excluded'])} excluded from {result['scanned']} candidates. Original claims unverified.")
    papers = []
    for record in result["matches"]:
        doc, meta, dist = record["document"], record["metadata"], record["distance"]
        papers.append({
            "arxiv_id": meta["arxiv_id"], "title": meta.get("title", ""),
            "categories": meta.get("categories", ""), "published": meta.get("published", ""),
            "vault_path": meta["vault_path"], "relevance_score": round(1 - dist, 3),
            "excerpt": doc[:400], "document": doc, "eligibility": record["eligibility"],
            "query_status": result["status"],
        })
    return papers


def cmd_search(cfg: dict, query: str) -> None:
    print(f'\nSearching vault: "{query}"')
    papers = do_search(cfg, query)
    if not papers:
        return
    print(f"\nFound {len(papers)} papers:\n")
    for p in papers:
        print_paper(
            f"{p['relevance_score']:.2f}",
            p["title"],
            p["arxiv_id"],
            p["published"],
            p["categories"],
            p["excerpt"],
        )
    print()


# ── Alpha ideas via Claude ────────────────────────────────────────────────────


def cmd_alpha_ideas(cfg: dict, topic: str) -> None:
    print(f'\nGenerating alpha ideas for: "{topic}"')
    print("Searching vault for relevant papers...")
    papers = do_search(cfg, topic, n=8)
    if not papers:
        print("No papers found in vault. Run `python run.py` first.")
        return

    paper_summaries = []
    for i, p in enumerate(papers, 1):
        content = p["document"][:2000]
        paper_summaries.append(
            f"--- Paper {i}: {p['title']} ({p['arxiv_id']}) ---\n{content}"
        )

    prompt = (
        f"You are a quant researcher specializing in systematic trading strategies.\n"
        f"Based on the following {len(papers)} local reading notes (original claims unverified), generate 3 specific, "
        f"implementable alpha ideas for the topic: '{topic}'.\n\n"
        f"For each idea include:\n"
        f"1. Signal construction (exact steps)\n"
        f"2. Data required (source, frequency)\n"
        f"3. Expected holding period\n"
        f"4. Key risk and failure mode\n\n"
        f"{'=' * 40}\n\n" + "\n\n".join(paper_summaries)
    )

    claude_bin = shutil.which("claude")
    if not claude_bin:
        print(
            "ERROR: 'claude' CLI not found. Install it via: npm install -g @anthropic-ai/claude-code"
        )
        return

    model = cfg.get("claude_model", "claude-haiku-4-5-20251001")
    print(f"\nAsking Claude ({model}) for alpha ideas...\n")
    print(DIVIDER)

    result = subprocess.run(
        [claude_bin, "--output-format", "text", "--model", model, "-p", "-"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=300,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        print(f"Claude CLI error: {result.stderr[:300]}")
        return

    print(result.stdout.strip())
    print(DIVIDER)


# ── Stats ─────────────────────────────────────────────────────────────────────


def cmd_stats(cfg: dict) -> None:
    conn = get_db(cfg)
    collection = get_collection(cfg)

    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    processed = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE processed = 1"
    ).fetchone()[0]
    abstract_only = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE processed = 1 AND vault_path IS NOT NULL"
    ).fetchone()[0]

    date_range = conn.execute(
        "SELECT MIN(published), MAX(published) FROM papers"
    ).fetchone()
    min_date = (date_range[0] or "")[:10]
    max_date = (date_range[1] or "")[:10]

    index_size = collection.count()

    # Papers per year (last 5)
    year_rows = conn.execute("""
        SELECT substr(published, 1, 4) as yr, COUNT(*) as cnt
        FROM papers
        WHERE yr >= cast(strftime('%Y', 'now') - 5 as text)
        GROUP BY yr ORDER BY yr DESC
    """).fetchall()

    # Top 5 categories
    all_cats: dict[str, int] = {}
    for (cats_json,) in conn.execute("SELECT categories FROM papers").fetchall():
        import json

        try:
            cats = json.loads(cats_json)
        except Exception:
            continue
        for c in cats:
            all_cats[c] = all_cats.get(c, 0) + 1
    top_cats = sorted(all_cats.items(), key=lambda x: -x[1])[:5]

    conn.close()

    print("\n" + DIVIDER)
    print("  QUANT VAULT STATISTICS")
    print(DIVIDER)
    print(f"  Total papers in DB    : {total:,}")
    print(f"  Processed (any)       : {processed:,}")
    print(f"  With vault file       : {abstract_only:,}")
    print(f"  ChromaDB index size   : {index_size:,}")
    print(f"  Date range            : {min_date} to {max_date}")
    print()
    print("  Papers per year (last 5):")
    for yr, cnt in year_rows:
        bar = "#" * min(cnt // 50, 40)
        print(f"    {yr}  {cnt:>5,}  {bar}")
    print()
    print("  Top 5 categories:")
    for cat, cnt in top_cats:
        print(f"    {cat:<15}  {cnt:>5,}")
    print(DIVIDER)


# ── Related papers ────────────────────────────────────────────────────────────


def cmd_related(cfg: dict, arxiv_id: str) -> None:
    conn = get_db(cfg)
    row = conn.execute(
        "SELECT title, abstract FROM papers WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchone()
    conn.close()

    if not row:
        print(f"Paper {arxiv_id} not found in DB.")
        return

    title, abstract = row
    print(f"\nFinding papers related to: {title}")
    query = f"{title} {abstract[:300]}"
    papers = do_search(cfg, query, n=8)

    # Filter out the paper itself
    papers = [p for p in papers if p["arxiv_id"] != arxiv_id][:6]
    if not papers:
        print("No related papers found.")
        return

    print(f"\nTop {len(papers)} related papers:\n")
    for p in papers:
        print_paper(
            f"{p['relevance_score']:.2f}",
            p["title"],
            p["arxiv_id"],
            p["published"],
            p["categories"],
            p["excerpt"],
        )
    print()


# ── Daily papers ──────────────────────────────────────────────────────────────


def cmd_daily(cfg: dict) -> None:
    conn = get_db(cfg)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    rows = conn.execute(
        """
        SELECT arxiv_id, title, categories, published, abstract
        FROM papers
        WHERE substr(fetched_at, 1, 10) >= ?
        ORDER BY published DESC
        LIMIT 50
    """,
        (yesterday,),
    ).fetchall()
    conn.close()

    if not rows:
        print(f"No new papers fetched today ({today}).")
        return

    print(f"\n{len(rows)} papers fetched in the last 24 hours:\n")
    for r in rows:
        import json

        try:
            cats = ", ".join(json.loads(r[2]))
        except Exception:
            cats = r[2]
        print_paper("NEW", r[1], r[0], r[3], cats, r[4])
    print()


# ── Top crypto papers ─────────────────────────────────────────────────────────


def _extract_crypto_score(content: str) -> float:
    """Extract numeric crypto applicability score from vault file."""
    # Look for "Crypto Applicability\nRate X/5" pattern
    match = re.search(r"Crypto.*?Rate\s+(\d)[/ ]5", content, re.IGNORECASE | re.DOTALL)
    if match:
        return float(match.group(1))
    # Also look for "Crypto: X/5" in relevance score line
    match = re.search(r"Crypto:\s*(\d)[/ ]5", content, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0


def cmd_top_crypto(cfg: dict, n: int) -> None:
    conn = get_db(cfg)
    rows = conn.execute("""
        SELECT arxiv_id, title, categories, published, vault_path
        FROM papers
        WHERE processed = 1 AND vault_path IS NOT NULL
        ORDER BY published DESC
        LIMIT 500
    """).fetchall()
    conn.close()

    scored = []
    for r in rows:
        note = current_note({"arxiv_id": r[0], "vault_path": r[4]}, source_root=cfg["vault_path"])
        score = _extract_crypto_score(note["document"]) if note else 0.0
        if score > 0:
            scored.append((score, r))

    scored.sort(key=lambda x: -x[0])
    top = scored[:n]

    if not top:
        print("No crypto-scored papers found (need Phase 2 processing).")
        return

    import json

    print(f"\nTop {len(top)} papers by Crypto Applicability score:\n")
    for score, r in top:
        try:
            cats = ", ".join(json.loads(r[2]))
        except Exception:
            cats = r[2]
        print_paper(f"Crypto {score:.0f}/5", r[1], r[0], r[3], cats, "")
    print()


# ── Export to markdown ────────────────────────────────────────────────────────


def cmd_export(cfg: dict, query: str, output_file: str) -> None:
    papers = do_search(cfg, query, n=20)
    if not papers:
        print("No papers found to export.")
        return

    lines = [
        f"# Quant Vault Export: {query}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"Results: {len(papers)} papers",
        "",
        "---",
        "",
    ]
    for p in papers:
        lines.append(f"## [{p['relevance_score']:.2f}] {p['title']}")
        lines.append("")
        lines.append(
            f"- **arXiv:** [{p['arxiv_id']}](https://arxiv.org/abs/{p['arxiv_id']})"
        )
        lines.append(f"- **Published:** {p['published'][:10]}")
        lines.append(f"- **Categories:** {p['categories']}")
        lines.append("")

        content = p["document"]
        if content.startswith("---"):
            end = content.find("---", 3)
            content = content[end + 3:].strip() if end > 0 else content
        lines.append(f"Source SHA-256: `{p['eligibility']['source_sha256']}`; "
                     f"query {p['query_status']}; local note, original claims unverified.")
        lines.append(content)
        lines.append("")
        lines.append("---")
        lines.append("")

    out_path = Path(output_file)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported {len(papers)} papers to {out_path}")


# ── Distill methodology ───────────────────────────────────────────────────────

_DISTILL_TOPICS = [
    # ── Foundational methodology ───────────────────────────────────────────────
    (
        "idea_generation",
        "how researchers generate trading strategy ideas, hypothesis formation, "
        "identifying anomalies, cross-disciplinary methods, data-driven discovery",
    ),
    (
        "mathematical_verification",
        "statistical testing methodology, information coefficient, t-statistics, "
        "bootstrap permutation test, multiple testing correction, deflated Sharpe, "
        "out-of-sample validation, p-hacking",
    ),
    (
        "signal_construction",
        "signal construction, feature engineering, cross-sectional ranking, "
        "winsorization, normalization, look-ahead bias prevention, alternative data",
    ),
    (
        "backtesting",
        "backtesting methodology, transaction costs modeling, turnover, Sharpe ratio, "
        "survivorship bias, point-in-time data, walk-forward, regime conditioning",
    ),
    (
        "failure_modes",
        "overfitting strategies, data mining bias, regime change, crowding, "
        "strategy failure modes, alpha decay, why strategies stop working",
    ),
    # ── Signal classes ─────────────────────────────────────────────────────────
    (
        "momentum_trend",
        "momentum factor, time series momentum, cross-sectional momentum, "
        "trend following, CTA, 12-1 momentum, intermediate-horizon momentum",
    ),
    (
        "mean_reversion",
        "mean reversion, pairs trading, statistical arbitrage, cointegration, "
        "Ornstein-Uhlenbeck, convergence trading, basis trading, spread trading",
    ),
    (
        "volatility_trading",
        "volatility forecasting, realized volatility, GARCH, HAR model, implied vol, "
        "vol surface, variance risk premium, volatility regime, VIX",
    ),
    (
        "market_microstructure",
        "market microstructure, order flow imbalance, Kyle lambda, price impact, "
        "bid-ask spread, Amihud illiquidity, high frequency trading, execution",
    ),
    (
        "factor_models",
        "factor model construction, Fama-French, risk premia, factor zoo, "
        "multi-factor portfolio, factor timing, factor exposure, smart beta",
    ),
    # ── Market-specific ────────────────────────────────────────────────────────
    (
        "crypto_defi",
        "cryptocurrency trading, Bitcoin, Ethereum, DeFi, on-chain data, "
        "perpetual futures, funding rate, crypto momentum, altcoin, blockchain analytics",
    ),
    (
        "hk_asian_equity",
        "Hong Kong equities, HKEX, Hang Seng, H-shares, A-shares, China market, "
        "Asian market microstructure, ADR premium, southbound northbound connect",
    ),
    # ── Domain cross-overs ─────────────────────────────────────────────────────
    (
        "econophysics",
        "econophysics power law fat tail Zipf Pareto Levy Hurst exponent long memory "
        "multifractal agent-based model financial network log-periodic market crash "
        "scaling law self-organized criticality return distribution",
    ),
    (
        "reinforcement_learning_trading",
        "reinforcement learning trading deep RL Q-learning policy gradient DDPG PPO "
        "actor-critic trading agent portfolio optimization order execution reward shaping "
        "market simulation multi-agent adaptive markets",
    ),
]


_PLAN_MODEL = "claude-opus-4-6"
_WRITE_MODEL = "claude-sonnet-4-6"

# Track active subprocess so Ctrl+C can kill it cleanly
_distill_proc: "subprocess.Popen | None" = None


def _run_claude(
    claude_bin: str, model: str, prompt: str, timeout: int = 300
) -> str | None:
    """Run claude CLI as a subprocess. One at a time — sequential, never parallel."""
    global _distill_proc
    proc = subprocess.Popen(
        [claude_bin, "--output-format", "text", "--model", model, "-p", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _distill_proc = proc
    try:
        stdout, stderr = proc.communicate(
            input=prompt.encode("utf-8", errors="replace"), timeout=timeout
        )
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        print(f"  Claude timeout ({model}) after {timeout}s")
        return None
    finally:
        _distill_proc = None
    if proc.returncode != 0:
        print(
            f"  Claude error ({model}): {stderr[:200].decode('utf-8', errors='replace')}"
        )
        return None
    return stdout.decode("utf-8", errors="replace").strip()


def cmd_distill(cfg: dict, output_path: str | None) -> None:
    """Distill quant research methodology from vault papers via Claude.

    Pipeline: Opus plans the synthesis outline → Sonnet writes each section.
    """
    claude_bin = shutil.which("claude")
    if not claude_bin:
        print("ERROR: 'claude' CLI not found.")
        return

    collection = get_collection(cfg)
    sections: list[str] = []
    used_sources: set[str] = set()

    print(DIVIDER)
    print("  DISTILLING QUANT RESEARCH METHODOLOGY FROM VAULT  (14 topics)")
    print(f"  Plan: {_PLAN_MODEL}  |  Write: {_WRITE_MODEL}")
    print(DIVIDER)

    for section_name, topic_query in _DISTILL_TOPICS:
        print(f"\n[{section_name}] Searching vault...", flush=True)
        results = query_note_sources(collection, topic_query, 10, source_root=cfg["vault_path"])
        print(f"  {results['status']}: eligible local notes; original claims unverified.")
        if not results["matches"]:
            continue

        excerpts = []
        for record in results["matches"]:
            doc, meta = record["document"], record["metadata"]
            used_sources.add(meta["arxiv_id"])
            title = meta.get("title", "Unknown")
            arxiv_id = meta.get("arxiv_id", "")
            excerpt = doc[:800].replace("\n", " ")
            excerpts.append(f"[{title} / {arxiv_id}]\n{excerpt}")

        paper_block = "\n\n---\n\n".join(excerpts)
        section_title = section_name.replace("_", " ").title()

        # ── Step 1: Opus plans the synthesis outline ──────────────────────────
        print("  Opus planning outline...", flush=True)
        plan_prompt = (
            f"You are a senior quantitative researcher. Given these local reading-note "
            f"excerpts (original claims unverified) on '{section_title}', create a concise synthesis outline.\n\n"
            f"Output a bullet-point outline (8-12 points) covering:\n"
            f"- Core principles from the literature\n"
            f"- Specific techniques/formulas worth including\n"
            f"- Key pitfalls these papers warn about\n"
            f"- Practical implementation steps\n\n"
            f"Be specific — note arXiv IDs next to claims they support. "
            f"This outline will guide a 400-word write-up.\n\n"
            f"{'=' * 40}\n\n{paper_block}"
        )
        outline = _run_claude(claude_bin, _PLAN_MODEL, plan_prompt, timeout=120)
        if not outline:
            outline = "(no outline generated)"

        # ── Step 2: Sonnet writes the section from outline + excerpts ─────────
        print("  Sonnet writing section...", flush=True)
        write_prompt = (
            f"Write a 400-word methodology section on "
            f"**{section_title}** in quantitative finance.\n\n"
            f"RULES:\n"
            f"- Output ONLY markdown. No intro sentence like 'Here is...'.\n"
            f"- Start directly with the content.\n"
            f"- Use this structure: Core Principle / Techniques / Pitfalls / Implementation\n"
            f"- Cite arXiv IDs inline (e.g. [0707.0385]) where the outline maps them.\n\n"
            f"OUTLINE TO FOLLOW:\n{outline}\n\n"
            f"PAPER EXCERPTS FOR GROUNDING:\n{'=' * 40}\n\n{paper_block}"
        )
        content = _run_claude(claude_bin, _WRITE_MODEL, write_prompt, timeout=300)
        if not content:
            continue

        sections.append(f"## {section_title}\n\n{content}")
        print("  Done.")

    if not sections:
        print("No sections generated.")
        return

    vault_path = cfg.get("vault_path", ".")
    out_file = output_path or str(
        Path(vault_path) / "guidelines" / "quant-methodology-distilled.md"
    )
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    content = (
        "# Quant Research Methodology — Distilled from Vault\n\n"
        "> Auto-generated from arXiv quant-finance corpus via `python research.py --distill`\n"
        f"> Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n"
        f"> Based on {len(used_sources)} eligible local paper notes; original claims unverified\n\n"
        "---\n\n" + "\n\n---\n\n".join(sections)
    )
    Path(out_file).write_text(content, encoding="utf-8")
    print(f"\n{DIVIDER}")
    print(f"  Distilled methodology saved to: {out_file}")
    print(DIVIDER)


# ── Snapshot ──────────────────────────────────────────────────────────────────


def cmd_snapshot(cfg: dict, topic: str) -> None:
    """Synthesize vault knowledge on a specific topic into a focused doc.

    Council decision: vault-first (no live fetch), single Sonnet call.
    Cost: ~$0.20 per run. Use --deepdive (future) for new-paper fetching.
    """
    claude_bin = shutil.which("claude")
    if not claude_bin:
        print("ERROR: 'claude' CLI not found.")
        return

    collection = get_collection(cfg)

    import re

    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    date_str = datetime.now().strftime("%Y-%m-%d")

    print(DIVIDER)
    print(f"  SNAPSHOT: {topic}")
    print("  Searching vault for top 12 relevant papers...")
    print(DIVIDER)

    results = query_note_sources(collection, topic, 12, source_root=cfg["vault_path"])
    print(f"  {results['status']}: local reading notes; original claims unverified.")
    if not results["matches"]:
        print(f"No papers found for '{topic}'. Try a broader query.")
        return

    excerpts = []
    for record in results["matches"]:
        doc, meta = record["document"], record["metadata"]
        title = meta.get("title", "Unknown")
        arxiv_id = meta.get("arxiv_id", "")
        published = (meta.get("published") or "")[:7]
        excerpt = doc[:1200].replace("\n", " ")
        excerpts.append(f"**[{arxiv_id}]** {title} ({published})\n{excerpt}")

    paper_block = "\n\n---\n\n".join(excerpts)

    prompt = (
        f"You are a senior quantitative researcher. Synthesize the following vault excerpts "
        f"on the topic: **{topic}**. These are local notes; original paper claims are unverified.\n\n"
        f"Output ONLY markdown with this structure (no preamble):\n\n"
        f"## Key Findings\n"
        f"3-5 bullet points: the most important, actionable insights from these papers.\n\n"
        f"## Signal / Strategy Ideas\n"
        f"What trading signals or strategies emerge from this research? Be specific.\n\n"
        f"## Implementation Notes\n"
        f"What does a practitioner need to know to implement the strongest idea here?\n"
        f"Include: data requirements, key parameters, failure modes to watch.\n\n"
        f"## Contradictions & Open Questions\n"
        f"Where do papers disagree? What's unresolved?\n\n"
        f"## Further Reading\n"
        f"List the 3 most important papers from below (arXiv ID + title).\n\n"
        f"{'=' * 50}\n\n{paper_block}"
    )

    print(f"  Sonnet synthesizing {len(results['matches'])} papers...", flush=True)
    content = _run_claude(claude_bin, _WRITE_MODEL, prompt, timeout=180)
    if not content:
        print("Synthesis failed.")
        return

    vault_path = cfg.get("vault_path", ".")
    out_dir = Path(vault_path) / "guidelines" / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{slug}-{date_str}.md"

    doc = (
        f"# Snapshot: {topic}\n\n"
        f"> Generated: {date_str} | Papers: {len(results['matches'])} | "
        f"Model: {_WRITE_MODEL}\n\n"
        f"---\n\n{content}\n"
    )
    out_file.write_text(doc, encoding="utf-8")

    print(f"\n{DIVIDER}")
    print(f"  Snapshot saved: {out_file}")
    print(f"  Use in Claude Code: @{out_file}")
    print("  (Run `python sync.py` to make it searchable via MCP)")
    print(DIVIDER)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quant Research Vault — CLI research tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python research.py "momentum factor crypto"
  python research.py --alpha-ideas "pairs trading"
  python research.py --stats
  python research.py --related 2401.12345
  python research.py --daily
  python research.py --top-crypto 10
  python research.py --export "momentum" out.md
  python research.py --distill
  python research.py --distill path/to/output.md
  python research.py --snapshot "crypto funding rate RL"
        """,
    )
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument(
        "--alpha-ideas",
        metavar="TOPIC",
        help="Generate alpha ideas for a topic via Claude",
    )
    parser.add_argument("--stats", action="store_true", help="Show vault statistics")
    parser.add_argument("--related", metavar="ARXIV_ID", help="Find related papers")
    parser.add_argument("--daily", action="store_true", help="Show today's new papers")
    parser.add_argument(
        "--top-crypto",
        type=int,
        metavar="N",
        help="Top N papers by crypto applicability score",
    )
    parser.add_argument(
        "--export",
        nargs=2,
        metavar=("QUERY", "OUTPUT"),
        help="Export search results to a markdown file",
    )
    parser.add_argument(
        "--distill",
        nargs="?",
        const="",
        metavar="OUTPUT",
        help="Distill methodology guide from vault (optional output path)",
    )
    parser.add_argument(
        "--snapshot",
        metavar="TOPIC",
        help="Synthesize vault knowledge on a specific topic into a focused doc",
    )
    args = parser.parse_args()

    # Kill active claude subprocess on Ctrl+C
    def _sigint(sig, frame):
        if _distill_proc and _distill_proc.poll() is None:
            print("\n[research.py] Ctrl+C — terminating Claude process...", flush=True)
            _distill_proc.terminate()
            try:
                _distill_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                _distill_proc.kill()
        sys.exit(1)

    signal.signal(signal.SIGINT, _sigint)

    cfg = load_config()

    if args.stats:
        cmd_stats(cfg)
    elif args.alpha_ideas:
        cmd_alpha_ideas(cfg, args.alpha_ideas)
    elif args.related:
        cmd_related(cfg, args.related)
    elif args.daily:
        cmd_daily(cfg)
    elif args.top_crypto:
        cmd_top_crypto(cfg, args.top_crypto)
    elif args.export:
        cmd_export(cfg, args.export[0], args.export[1])
    elif args.distill is not None:
        cmd_distill(cfg, args.distill or None)
    elif args.snapshot:
        cmd_snapshot(cfg, args.snapshot)
    elif args.query:
        cmd_search(cfg, args.query)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
