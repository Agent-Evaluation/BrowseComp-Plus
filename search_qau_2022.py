from pyserini.search.lucene import LuceneSearcher
import json

searcher = LuceneSearcher('indexes/bm25')
hits = searcher.search('Queen Arwa University', k=200)

for hit in hits:
    raw = json.loads(hit.lucene_document.get('raw'))
    date = raw.get('date', '')
    if '2022' in date:
        title = raw.get('title', 'NO TITLE')[:100]
        print(f'[{hit.docid}] {date} - {title}')
