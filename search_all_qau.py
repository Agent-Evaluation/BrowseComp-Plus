from pyserini.search.lucene import LuceneSearcher
import json

searcher = LuceneSearcher('indexes/bm25')
hits = searcher.search('Queen Arwa', k=300)

print("All Queen Arwa University articles:")
for hit in hits[:50]:
    raw = json.loads(hit.lucene_document.get('raw'))
    date = raw.get('date', 'NO DATE')
    title = raw.get('title', 'NO TITLE')[:100]
    print(f'[{hit.docid}] {date} - {title}')
