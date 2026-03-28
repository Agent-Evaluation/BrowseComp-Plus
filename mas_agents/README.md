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

| Tool | Install |
|---|---|
| **Python 3.10 + uv** | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Bun** | `curl -fsSL https://bun.sh/install \| bash` |
| **Java 21** | macOS: `brew install openjdk@21` · Windows: [Microsoft JDK 21](https://learn.microsoft.com/en-us/java/openjdk/download) |
| **GitHub Copilot CLI** | Must be installed and authenticated (`gh auth login`) |
| **HuggingFace CLI** | `pip install huggingface-cli` then `huggingface-cli login` |

## Quick Start (Fresh Clone)

Everything below runs from the repo root. Works on **macOS** and **Windows** (PowerShell).

```bash
git clone https://github.com/Agent-Evaluation/BrowseComp-Plus.git
cd BrowseComp-Plus
```

### 1. Install Python dependencies
```bash
uv sync
```

### 2. Install TypeScript dependencies
```bash
cd mas_agents && bun install && cd ..
```

### 3. Download & decrypt the dataset
```bash
huggingface-cli login
uv run python scripts_build_index/decrypt_dataset.py \
  --output data/browsecomp_plus_decrypted.jsonl \
  --generate-tsv topics-qrels/queries.tsv
```
This generates `data/browsecomp_plus_decrypted.jsonl` (ground truth) and `topics-qrels/queries.tsv` (830 queries).

### 4. Download the BM25 index
```bash
bash scripts_build_index/download_indexes.sh
```
Or download just the BM25 index:
```bash
huggingface-cli download Tevatron/browsecomp-plus-indexes \
  --repo-type=dataset --include="bm25/*" --local-dir ./indexes
```

### 5. Set JAVA_HOME (if not auto-detected)
```bash
# macOS (Homebrew)
export JAVA_HOME=$(/usr/libexec/java_home -v 21)

# Windows (PowerShell)
$env:JAVA_HOME = "C:\Program Files\Microsoft\jdk-21.0.10.7-hotspot"
```

### 6. Run!
```bash
bun run mas_agents/src/run-eval.ts \
  --architecture single --model gpt-4.1 \
  --index-path indexes/bm25/ --limit 1
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
