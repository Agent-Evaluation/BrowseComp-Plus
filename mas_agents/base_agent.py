"""
Abstract base class for BrowseComp-Plus Copilot-powered agents.

Adapted from plancraft/agents/copilot_base.py:
  - Replaces Minecraft oracle search with BM25 retriever (Pyserini)
  - Tracks retrieved_docids and tool_call_counts for BrowseComp-Plus output format
  - Keeps same Copilot SDK interface and conversation management
"""

import json
import re
from abc import ABC, abstractmethod

from copilot import CopilotClient
from transformers import AutoTokenizer


class RetrieverTool:
    """
    Local BM25 retriever wrapping Pyserini LuceneSearcher.

    Matches the paper's configuration:
      - top-k = 5 results per query
      - snippet_max_tokens = 512 (Qwen3-0.6B tokenizer)
    """

    def __init__(self, index_path: str, k: int = 5, snippet_max_tokens: int = 512):
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(index_path)
        self.k = k
        self.snippet_max_tokens = snippet_max_tokens
        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    def search(self, query: str) -> list[dict]:
        """
        Search the BM25 index and return top-k hits with truncated snippets.

        Returns:
            List of dicts with keys: docid, score, snippet
        """
        hits = self.searcher.search(query, self.k)
        results = []

        for hit in hits:
            raw = json.loads(hit.lucene_document.get("raw"))
            text = raw["contents"]

            # Truncate to snippet_max_tokens
            if self.tokenizer and self.snippet_max_tokens > 0:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > self.snippet_max_tokens:
                    text = self.tokenizer.decode(
                        tokens[: self.snippet_max_tokens],
                        skip_special_tokens=True,
                    )

            results.append({
                "docid": hit.docid,
                "score": hit.score,
                "snippet": text,
            })

        return results

    def format_results(self, results: list[dict]) -> str:
        """Format search results into a readable string for the agent."""
        if not results:
            return "No results found."

        parts = []
        for r in results:
            parts.append(
                f"[DocID: {r['docid']}] (score: {r['score']:.4f})\n{r['snippet']}"
            )
        return "\n\n---\n\n".join(parts)


class CopilotBaseAgent(ABC):
    """
    Abstract Base Class for Copilot-powered BrowseComp-Plus agents.
    """

    def __init__(self, model_name: str, client: CopilotClient, retriever: RetrieverTool):
        self.model_name = model_name
        self.client = client
        self.retriever = retriever
        self.conversation: list[dict] = []
        self.step_count = 0
        self.query_id: str | None = None
        self.query_text: str | None = None
        # Tracking for output
        self.retrieved_docids: set[str] = set()
        self.tool_call_counts: dict[str, int] = {"search": 0}

    def reset(self, query_id: str, query_text: str):
        """Resets the agent state for a new query."""
        self.conversation = []
        self.step_count = 0
        self.query_id = query_id
        self.query_text = query_text
        self.retrieved_docids = set()
        self.tool_call_counts = {"search": 0}

    @abstractmethod
    async def act(self, context: str | None) -> str:
        """
        Receives optional context and returns agent output.
        Async because the Copilot SDK is async.
        """
        pass

    def _execute_search(self, query: str) -> str:
        """
        Executes a retriever search and tracks results.

        Args:
            query: Search query string

        Returns:
            Formatted search results string
        """
        results = self.retriever.search(query)
        self.tool_call_counts["search"] = self.tool_call_counts.get("search", 0) + 1

        for r in results:
            self.retrieved_docids.add(str(r["docid"]))

        return self.retriever.format_results(results)

    def _extract_search_query(self, text: str) -> str | None:
        """Extract search query from agent output if it contains a search action."""
        match = re.search(r"search:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _is_final_answer(self, text: str) -> bool:
        """Check if the agent's output contains a final answer."""
        text_lower = text.lower()
        has_exact_answer = "exact answer:" in text_lower
        has_search = bool(re.search(r"^search:\s*\S", text, re.IGNORECASE | re.MULTILINE))
        return has_exact_answer and not has_search

    def get_output(self) -> dict:
        """
        Build the BrowseComp-Plus output JSON for this query.
        """
        # Extract the final answer from the last model message
        final_output = ""
        for msg in reversed(self.conversation):
            if msg["role"] in ("model", "assistant"):
                final_output = msg["content"]
                break

        return {
            "query_id": self.query_id,
            "tool_call_counts": dict(self.tool_call_counts),
            "status": "completed",
            "retrieved_docids": list(self.retrieved_docids),
            "result": [
                {
                    "type": "output_text",
                    "output": final_output,
                }
            ],
        }

    def log(self, msg: str):
        print(f"[{self.__class__.__name__}] {msg}")
