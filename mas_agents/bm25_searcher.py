"""
Standalone BM25 search bridge for the TypeScript mas_agents.

Reads JSONL requests from stdin, writes JSONL responses to stdout.
Stays alive for multiple requests (one JSON object per line).

Request format:
    {"action": "search", "query": "...", "k": 5, "snippet_max_tokens": 512}

Response format:
    {"results": [{"docid": "...", "score": 1.23, "snippet": "..."}]}

Usage:
    python bm25_searcher.py --index-path indexes/bm25/
"""

import json
import sys
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", required=True)
    args = parser.parse_args()

    from pyserini.search.lucene import LuceneSearcher
    from transformers import AutoTokenizer

    searcher = LuceneSearcher(args.index_path)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    # Signal ready
    sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
            sys.stdout.flush()
            continue

        action = req.get("action")
        if action == "search":
            query = req.get("query", "")
            k = req.get("k", 5)
            snippet_max_tokens = req.get("snippet_max_tokens", 512)

            hits = searcher.search(query, k)
            results = []

            for hit in hits:
                raw = json.loads(hit.lucene_document.get("raw"))
                text = raw["contents"]

                if snippet_max_tokens and snippet_max_tokens > 0:
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    if len(tokens) > snippet_max_tokens:
                        text = tokenizer.decode(
                            tokens[:snippet_max_tokens],
                            skip_special_tokens=True,
                        )

                results.append({
                    "docid": hit.docid,
                    "score": float(hit.score),
                    "snippet": text,
                })

            sys.stdout.write(json.dumps({"results": results}) + "\n")
            sys.stdout.flush()

        elif action == "quit":
            break

        else:
            sys.stdout.write(json.dumps({"error": f"Unknown action: {action}"}) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
