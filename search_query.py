#!/usr/bin/env python3
"""Simple search script for BrowseComp-Plus knowledge base."""

import json
import sys
from pyserini.search.lucene import LuceneSearcher

def search(query: str, k: int = 10):
    """Search the BM25 index and return results."""
    index_path = "indexes/bm25"
    searcher = LuceneSearcher(index_path)
    
    hits = searcher.search(query, k)
    results = []
    
    for i, hit in enumerate(hits):
        raw = json.loads(hit.lucene_document.get("raw"))
        results.append({
            "rank": i + 1,
            "docid": hit.docid,
            "score": hit.score,
            "text": raw["contents"]
        })
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search_query.py <query>")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    print(f"Searching for: {query}\n")
    
    results = search(query, k=10)
    
    for result in results:
        print(f"[{result['docid']}] Score: {result['score']:.4f}")
        print(f"Text: {result['text'][:500]}...")
        print("-" * 80)
