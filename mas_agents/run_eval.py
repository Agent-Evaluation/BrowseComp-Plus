"""
BrowseComp-Plus multi-agent evaluation runner.

Runs a single architecture against all 100 queries and saves JSON outputs
in the BrowseComp-Plus format under runs/<run_name>/.

Usage:
    uv run python mas_agents/run_eval.py \
        --architecture single \
        --model gpt-5-mini \
        --index-path indexes/bm25/ \
        --output-dir runs/bm25/single_gpt5mini
"""

import argparse
import asyncio
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm


async def main():
    parser = argparse.ArgumentParser(
        description="Run BrowseComp-Plus evaluation with MAS architectures"
    )
    parser.add_argument(
        "--architecture",
        choices=["single", "independent", "centralized", "decentralized", "hybrid"],
        required=True,
        help="Agent architecture to evaluate",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="Copilot model name (default: %(default)s)",
    )
    parser.add_argument(
        "--query",
        default="topics-qrels/queries.tsv",
        help="Path to queries TSV file (default: %(default)s)",
    )
    parser.add_argument(
        "--index-path",
        default="indexes/bm25/",
        help="Path to BM25 Lucene index (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save results (default: runs/bm25/<architecture>_<model>)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of search results per query (default: %(default)s)",
    )
    parser.add_argument(
        "--snippet-max-tokens",
        type=int,
        default=512,
        help="Max tokens per document snippet (default: %(default)s)",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=3,
        help="Number of sub-agents for MAS architectures (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of queries to process (0=all, default: %(default)s)",
    )

    args = parser.parse_args()

    # Resolve output directory
    if args.output_dir is None:
        model_slug = args.model.replace("/", "_").replace("-", "_")
        args.output_dir = f"runs/bm25/{args.architecture}_{model_slug}"

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load queries ──────────────────────────────────────────────────────────
    query_path = Path(args.query)
    if not query_path.is_file():
        print(f"Error: Query file not found: {query_path}")
        sys.exit(1)

    queries: list[tuple[str, str]] = []
    with query_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                queries.append((row[0].strip(), row[1].strip()))

    print(f"Loaded {len(queries)} queries from {query_path}")

    if args.limit > 0:
        queries = queries[: args.limit]
        print(f"  Limited to {args.limit} queries")

    # ── Resume: skip already-processed queries ────────────────────────────────
    processed_ids: set[str] = set()
    if out_dir.exists():
        for json_path in out_dir.glob("run_*.json"):
            try:
                with json_path.open("r", encoding="utf-8") as jf:
                    meta = json.load(jf)
                    qid = meta.get("query_id")
                    if qid:
                        processed_ids.add(str(qid))
            except Exception:
                continue

    remaining = [(qid, qtext) for qid, qtext in queries if qid not in processed_ids]
    print(f"Processing {len(remaining)} remaining queries (skipping {len(processed_ids)} already done)")

    if not remaining:
        print("All queries already processed. Done.")
        return

    # ── Initialize retriever ──────────────────────────────────────────────────
    from .base_agent import RetrieverTool

    print(f"Initializing BM25 retriever from {args.index_path}...")
    retriever = RetrieverTool(
        index_path=args.index_path,
        k=args.k,
        snippet_max_tokens=args.snippet_max_tokens,
    )

    # ── Initialize Copilot client ─────────────────────────────────────────────
    from .llm_client import get_copilot_client

    print(f"Starting Copilot SDK client...")
    client = await get_copilot_client()

    # ── Create agent ──────────────────────────────────────────────────────────
    agent = _create_agent(args.architecture, args.model, client, retriever, args.num_agents)
    print(f"Agent: {agent.__class__.__name__} | Model: {args.model}")
    print(f"Output: {out_dir}")
    print("=" * 60)

    # ── Process queries ───────────────────────────────────────────────────────
    completed = 0
    failed = 0

    for qid, qtext in tqdm(remaining, desc="Queries", unit="query"):
        try:
            result = await agent.run_query(qid, qtext)

            # Save result
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            filename = out_dir / f"run_{ts}_{qid}.json"
            with filename.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)

            status = result.get("status", "unknown")
            search_count = result.get("tool_call_counts", {}).get("search", 0)
            print(f"  [{qid}] status={status} | searches={search_count}")

            if status == "completed":
                completed += 1
            else:
                failed += 1

        except Exception as exc:
            print(f"  [{qid}] FAILED: {exc}")
            failed += 1

            # Save failure record
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            filename = out_dir / f"run_{ts}_{qid}.json"
            with filename.open("w", encoding="utf-8") as f:
                json.dump({
                    "query_id": qid,
                    "tool_call_counts": {},
                    "status": f"error: {exc}",
                    "retrieved_docids": [],
                    "result": [{"type": "output_text", "output": ""}],
                }, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = completed + failed
    print("\n" + "=" * 60)
    print(f"Done! {completed}/{total} completed, {failed}/{total} failed")
    print(f"Results saved to: {out_dir}")

    # Stop Copilot client
    try:
        await client.stop()
    except Exception:
        pass


def _create_agent(architecture: str, model: str, client, retriever, num_agents: int):
    """Factory function to create the appropriate agent."""
    from .single_agent import CopilotSingleAgent
    from .multi_agents import (
        CopilotIndependentAgent,
        CopilotCentralizedAgent,
        CopilotDecentralizedAgent,
        CopilotHybridAgent,
    )

    if architecture == "single":
        return CopilotSingleAgent(model, client, retriever)
    elif architecture == "independent":
        return CopilotIndependentAgent(model, client, retriever, num_agents=num_agents)
    elif architecture == "centralized":
        return CopilotCentralizedAgent(model, client, retriever, num_agents=num_agents, rounds=5)
    elif architecture == "decentralized":
        return CopilotDecentralizedAgent(model, client, retriever, num_agents=num_agents, rounds=3)
    elif architecture == "hybrid":
        return CopilotHybridAgent(model, client, retriever, num_agents=num_agents, rounds=1, peer_rounds=1)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


if __name__ == "__main__":
    asyncio.run(main())
