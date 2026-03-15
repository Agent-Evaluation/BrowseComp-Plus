# BrowseComp-Plus Multi-Agent Evaluation (TypeScript / Bun)

Implements the 5 canonical agent architectures from Kim et al. (2025), *"Towards a Science of Scaling Agent Systems"*, adapted for BrowseComp-Plus web browsing tasks.

**Runtime**: [Bun](https://bun.sh/) + TypeScript  
**LLM client**: `@github/copilot-sdk` (npm)  
**Retriever**: BM25 via Pyserini (Python subprocess bridge)

## Architectures

| Architecture | Agents | Coordination | Paper Reference |
|---|---|---|---|
| **Single (SAS)** | 1 | None (baseline) | §3.1 |
| **Independent** | 3 | Synthesis-only | §3.1, Ω = synthesis_only |
| **Centralized** | 1 orch + 3 workers | Hierarchical, 5 rounds | §3.1, Ω = hierarchical |
| **Decentralized** | 3 | All-to-all debate, 3 rounds | §3.1, Ω = consensus |
| **Hybrid** | 1 orch + 3 workers | Hierarchical + lateral peer | §3.1, Ω = hierarchical + lateral |

## Prerequisites

1. **Bun**: Install from https://bun.sh/
2. **Python + uv**: Required for BM25 bridge (Pyserini)
3. **Java 21**: Required for Pyserini's Lucene index
4. **GitHub Copilot CLI**: Must be installed and authenticated
5. **Dataset**: Decrypt queries and download BM25 index (see below)

## Setup

```bash
cd BrowseComp-Plus

# Install Python deps (for BM25 bridge only)
uv sync

# Install TypeScript deps
cd mas_agents && bun install && cd ..

# Set JAVA_HOME (Windows)
$env:JAVA_HOME = "C:\Program Files\Microsoft\jdk-21.0.10.7-hotspot"

# Decrypt dataset (requires HuggingFace login)
huggingface-cli login
uv run python scripts_build_index/decrypt_dataset.py \
  --output data/browsecomp_plus_decrypted.jsonl \
  --generate-tsv topics-qrels/queries.tsv

# Download BM25 index
huggingface-cli download Tevatron/browsecomp-plus-indexes \
  --repo-type=dataset --include="bm25/*" --local-dir ./indexes
```

## Running a Single Architecture

```bash
bun run mas_agents/src/run-eval.ts \
  --architecture single \
  --model gpt-4.1 \
  --index-path indexes/bm25/ \
  --limit 1
```

Options:
- `--architecture`: `single`, `independent`, `centralized`, `decentralized`, `hybrid`
- `--model`: Any model available via GitHub Copilot SDK (default: `gpt-4.1`)
- `--limit N`: Process only the first N queries (useful for testing)
- `--num-agents`: Number of sub-agents for MAS architectures (default: 3)

## Running All 5 Architectures

```bash
bun run mas_agents/src/benchmark-runner.ts \
  --model gpt-4.1 \
  --index-path indexes/bm25/
```

This runs all 5 architectures sequentially and produces:
- `runs/bm25/<arch>_<model>/` — per-query JSON results
- `runs/BENCHMARK_REPORT.md` — markdown summary
- `runs/benchmark_summary.json` — machine-readable summary

## Evaluation (LLM-as-Judge)

Uses Kimi K2.5 via NVIDIA NIM as the judge model:

```bash
# Set API key
set NVIDIA_API_KEY=nvapi-YOUR_KEY_HERE

# Evaluate a single run
bun run mas_agents/src/evaluate-kimi.ts \
  --input-dir runs/bm25/single_gpt_4.1 \
  --ground-truth data/browsecomp_plus_decrypted.jsonl

# Or pass key directly
bun run mas_agents/src/evaluate-kimi.ts \
  --input-dir runs/bm25/single_gpt_4.1 \
  --api-key nvapi-YOUR_KEY_HERE
```

Output: `runs/bm25/single_gpt_4.1/eval_summary.json` with accuracy, calibration error, and per-query metrics.

## Output Format

Each query produces a JSON file compatible with BrowseComp-Plus evaluation:

```json
{
  "query_id": "42",
  "tool_call_counts": {"search": 5},
  "status": "completed",
  "retrieved_docids": ["doc_123", "doc_456"],
  "result": [
    {
      "type": "output_text",
      "output": "Explanation: ... Exact Answer: ... Confidence: 85%"
    }
  ]
}
```

## File Structure

```
mas_agents/
  package.json             # Bun project config
  tsconfig.json            # TypeScript config
  bm25_searcher.py         # Python BM25 bridge (JSONL stdin/stdout)
  src/
    index.ts               # Package re-exports
    bm25-bridge.ts         # Spawns Python process for BM25 search
    prompts.ts             # System + role-specific prompts
    llm-client.ts          # Copilot SDK wrapper with retry logic
    base-agent.ts          # Abstract base class + retriever integration
    single-agent.ts        # Single-Agent System (SAS)
    multi-agents.ts        # Independent, Centralized, Decentralized, Hybrid
    run-eval.ts            # Per-architecture query runner (CLI)
    benchmark-runner.ts    # Orchestrates all 5 architectures (CLI)
    evaluate-kimi.ts       # Kimi K2.5 LLM-as-judge evaluator (CLI)
  README.md                # This file
```
