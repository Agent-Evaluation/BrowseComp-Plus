#!/usr/bin/env python3
import json
from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher('indexes/bm25')

# Search for all documents and look for ones with archive pattern from 2022
hits = searcher.search('university', k=5000)

print("Looking for 2022 archive documents...")
count = 0
for i, hit in enumerate(hits):
    raw = json.loads(hit.lucene_document.get("raw"))
    contents = raw.get("contents", "")
    
    # Look for archive pattern with 2022
    if "archive" in contents.lower() and "2022" in contents:
        count += 1
        title = raw.get("title", "NO TITLE")[:150]
        date = raw.get("date", "NO DATE")
        print(f"\n[{hit.docid}] {date} - {title}")
        # Print the archive line
        for line in contents.split('\n'):
            if 'archive' in line.lower() and '2022' in line:
                print(f"Archive: {line.strip()}")
                break
        if count > 20:
            break
