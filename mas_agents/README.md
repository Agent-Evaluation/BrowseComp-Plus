# BrowseComp-Plus Multi-Agent Evaluation

Implements the 5 canonical agent architectures from Kim et al. (2025), *"Towards a Science of Scaling Agent Systems"*, adapted for BrowseComp-Plus web browsing tasks.

## Architectures

| Architecture | Agents | Coordination | Paper Reference |
|---|---|---|---|
| **Single (SAS)** | 1 | None (baseline) | §3.1 |
| **Independent** | 3 | Synthesis-only | §3.1, Ω = synthesis_only |
| **Centralized** | 1 orch + 3 workers | Hierarchical, 5 rounds | §3.1, Ω = hierarchical |
| **Decentralized** | 3 | All-to-all debate, 3 rounds | §3.1, Ω = consensus |
| **Hybrid** | 1 orch + 3 workers | Hierarchical + lateral peer | §3.1, Ω = hierarchical + lateral |

## Prerequisites

1. **Environment**: `uv sync` in the BrowseComp-Plus root
2. **Java 21**: Required for Pyserini BM25 index
3. **GitHub Copilot CLI**: Must be installed and authenticated
4. **Dataset**: Decrypt queries and download BM25 index (see below)

## Setup

```bash
cd BrowseComp-Plus

# Install dependencies
uv sync

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
uv run python -m mas_agents.run_eval \
  --architecture single \
  --model gpt-5-mini \
  --index-path indexes/bm25/ \
  --output-dir runs/bm25/single_gpt5mini
```

Options:
- `--architecture`: `single`, `independent`, `centralized`, `decentralized`, `hybrid`
- `--model`: Any model available via GitHub Copilot SDK
- `--limit N`: Process only the first N queries (useful for testing)
- `--num-agents`: Number of sub-agents for MAS architectures (default: 3)

## Running All 5 Architectures

```bash
uv run python -m mas_agents.benchmark_runner \
  --model gpt-5-mini \
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
uv run python -m mas_agents.evaluate_kimi \
  --input-dir runs/bm25/single_gpt5mini \
  --ground-truth data/browsecomp_plus_decrypted.jsonl

# Or pass key directly
uv run python -m mas_agents.evaluate_kimi \
  --input-dir runs/bm25/single_gpt5mini \
  --api-key nvapi-YOUR_KEY_HERE
```

Output: `runs/bm25/single_gpt5mini/eval_summary.json` with accuracy, calibration error, and per-query metrics.

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
  __init__.py              # Package exports
  base_agent.py            # Base class + BM25 retriever integration
  llm_client.py            # Copilot SDK wrapper with retry logic
  prompts.py               # System + role-specific prompts
  single_agent.py          # Single-Agent System (SAS)
  multi_agents.py          # Independent, Centralized, Decentralized, Hybrid
  run_eval.py              # Per-architecture query runner
  benchmark_runner.py      # Orchestrates all 5 architectures
  evaluate_kimi.py         # Kimi K2.5 LLM-as-judge evaluator
  README.md                # This file
```
