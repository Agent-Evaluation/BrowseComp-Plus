#!/usr/bin/env python3
import json
from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher('indexes/bm25')
hits = searcher.search('Queen Arwa 2022', k=100)

print("Articles from 2022 related to Queen Arwa:")
for i, hit in enumerate(hits):
    raw = json.loads(hit.lucene_document.get("raw"))
    date = raw.get("date", "NO DATE")
    if "2022" in date:
        title = raw.get("title", "NO TITLE")[:150]
        print(f"{i+1}. [{hit.docid}] {date} - {title}")
        print(f"   Text: {raw['contents'][:200]}...")
        print()
