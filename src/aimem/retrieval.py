import re

from rank_bm25 import BM25Okapi

from .schema import Instance


def units(inst: Instance) -> list[tuple[int, str]]:
    return [(s.index, f"{t.role}: {t.content}") for s in inst.sessions for t in s.turns]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


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


def sessions_of(inst: Instance, doc_ids: list[int]) -> list[int]:
    u = units(inst)
    out = []
    for d in doc_ids:
        s = u[d][0]
        if s not in out:
            out.append(s)
    return out
