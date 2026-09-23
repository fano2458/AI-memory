import re

from rank_bm25 import BM25Okapi

from .schema import Instance


def units(inst: Instance) -> list[tuple[int, str]]:
    return [(s.index, f"{t.role}: {t.content}") for s in inst.sessions for t in s.turns]


STOPWORDS = set(
    """a an and are as at be been being but by for from had has have he her his i if in into
    is it its me my of on or our she that the their them then there these they this to was we
    were what when which who will with you your am do does did doing ive im id youre thats
    just really some so very can could would should about after before other more most own
    same too only up out down over under again also get got make made take took like lot new
    bit dont arent""".split()
)


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


class Oracle:
    name = "oracle"

    def search(self, inst: Instance, query: str, k: int) -> list[int]:
        return [i for i, (sess, _) in enumerate(units(inst)) if sess in inst.gold_sessions]


class BM25:
    name = "bm25"

    def search(self, inst: Instance, query: str, k: int) -> list[int]:
        corpus = [tokenize(t) for _, t in units(inst)]
        scores = BM25Okapi(corpus).get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:k]


class Dense:
    name = "dense"

    def __init__(self, embedder):
        self.embedder = embedder

    def scores(self, texts: list[str], query: str):
        mat = self.embedder.get(texts)
        q = self.embedder.get([query])[0]
        return mat @ q

    def search(self, inst: Instance, query: str, k: int) -> list[int]:
        texts = [t for _, t in units(inst)]
        s = self.scores(texts, query)
        return sorted(range(len(s)), key=lambda i: s[i], reverse=True)[:k]


class Hybrid:
    name = "hybrid"

    def __init__(self, dense, rrf_k: int = 60):
        self.dense = dense
        self.rrf_k = rrf_k

    def fuse(self, rankings: list[list[int]], k: int) -> list[int]:
        score: dict[int, float] = {}
        for ranking in rankings:
            for rank, doc in enumerate(ranking):
                score[doc] = score.get(doc, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        return sorted(score, key=score.get, reverse=True)[:k]


def sessions_of(inst: Instance, doc_ids: list[int]) -> list[int]:
    u = units(inst)
    out = []
    for d in doc_ids:
        s = u[d][0]
        if s not in out:
            out.append(s)
    return out
