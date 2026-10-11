"""Hybrid retrieval: exact code match + BM25 + embeddings, fused with RRF."""
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

DATA = Path(__file__).parent / "data"
MODEL = "all-MiniLM-L6-v2"


def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())


@lru_cache(maxsize=1)
def _load():
    from sentence_transformers import SentenceTransformer

    items = json.loads((DATA / "items.json").read_text())
    emb = np.load(DATA / "embeddings.npy")
    docs = [tokenize(f'{r["section"]} {r["subheading"]} {r["description"]} {r["supplier_code"]}') for r in items]
    return items, emb, BM25Okapi(docs), SentenceTransformer(MODEL)


def search(query, k=8):
    items, emb, bm25, model = _load()
    q = query.strip()
    scores = {}

    def add(order, weight=1.0):
        for rank, i in enumerate(order):
            scores[i] = scores.get(i, 0.0) + weight / (60 + rank)

    # exact / prefix match on supplier code
    ql = q.lower()
    exact = [i for i, r in enumerate(items) if r["supplier_code"].lower() == ql]
    if exact:
        add(exact, 5.0)
    elif re.fullmatch(r"[\w-]{4,}", q) and any(c.isdigit() for c in q):
        add([i for i, r in enumerate(items) if r["supplier_code"].lower().startswith(ql)][:k], 2.0)

    toks = tokenize(q)
    if toks:
        add(np.argsort(-bm25.get_scores(toks))[:50])
    qe = model.encode([q], normalize_embeddings=True)[0]
    add(np.argsort(-(emb @ qe))[:50])

    top = sorted(scores, key=scores.get, reverse=True)[:k]
    return [items[i] for i in top]


STOP = {"a", "an", "the", "for", "of", "and", "show", "me", "list", "all", "every", "item", "items",
        "find", "get", "price", "prices", "code", "codes", "description", "with", "in", "any", "available"}


def search_all(query, fallback_k=50, keywords=None, max_price=None):
    """Return every row containing all meaningful query words (ranked by similarity);
    if none match, relax to rows containing any word, then to semantic top-k."""
    items, emb, bm25, model = _load()

    # a supplier code typed exactly (alone or inside a sentence) means the user wants that item only
    asked = {t.lower() for t in re.findall(r"[\w-]+", query) if any(c.isdigit() for c in t)}
    asked.add(query.strip().lower())
    exact = [r for r in items if r["supplier_code"].lower() in asked]
    if exact:
        return [r for r in exact if max_price is None or r["price"] <= max_price]

    toks = [t for t in (tokenize(" ".join(keywords)) if keywords else tokenize(query)) if t not in STOP]
    texts = [
        f'{r["section"]} {r["subheading"]} {r["description"]} {r["supplier_code"]}'.lower() for r in items
    ]
    qe = model.encode([query], normalize_embeddings=True)[0]
    sims = emb @ qe

    hits = []
    if toks:
        hits = [i for i, t in enumerate(texts) if all(w in t for w in toks)]
        if not hits:
            hits = [i for i, t in enumerate(texts) if any(w in t for w in toks)]
    if not hits:
        hits = list(np.argsort(-sims)[:fallback_k])
    # rank matches by fusing keyword (BM25), semantic and exact-code signals
    kw = bm25.get_scores(tokenize(query)) if tokenize(query) else np.zeros(len(items))
    ql = query.strip().lower()
    score = {i: 0.0 for i in hits}
    for order in (sorted(hits, key=lambda i: -kw[i]), sorted(hits, key=lambda i: -sims[i])):
        for rank, i in enumerate(order):
            score[i] += 1 / (10 + rank)
    for i in hits:
        code = items[i]["supplier_code"].lower()
        if code == ql:
            score[i] += 1.0
        elif ql and code.startswith(ql):
            score[i] += 0.1
    hits.sort(key=lambda i: -score[i])
    if max_price is not None:
        hits = [i for i in hits if items[i]["price"] <= max_price]
    return [items[i] for i in hits]
