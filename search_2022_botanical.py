#!/usr/bin/env python3
import json
from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher('indexes/bm25')
hits = searcher.search('2022 university science students field trip botanical samples', k=100)

print("2022 Articles about field trips/botanical samples:")
for i, hit in enumerate(hits):
    raw = json.loads(hit.lucene_document.get("raw"))
    date = raw.get("date", "NO DATE")
    if "2022" in date:
        title = raw.get("title", "NO TITLE")[:150]
        print(f"{i+1}. [{hit.docid}] {date} - {title}")
        print(f"   Text: {raw['contents'][:300]}...")
        print()
