import argparse
import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from rank_bm25 import BM25Okapi

from aimem import stale
from aimem.embed import Embedder
from aimem.retrieval import tokenize

KS = [1, 2, 4, 8, 16, 32]
METHODS = ["bm25", "dense", "hybrid"]
RRF_K = 60


def past_units(inst):
    return [(s.index, t.content) for s in inst.sessions if s.index < inst.new_session for t in s.turns]


def chance(n_pool, n_gold, k):
    misses = min(k, n_pool - n_gold)
    return 1 - math.prod((n_pool - n_gold - j) / (n_pool - j) for j in range(misses))


def ztest(a, b):
    p1, p2 = sum(a) / len(a), sum(b) / len(b)
    p = (sum(a) + sum(b)) / (len(a) + len(b))
    se = math.sqrt(p * (1 - p) * (1 / len(a) + 1 / len(b)))
    if not se:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def rrf(rankings, k):
    score = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            score[doc] = score.get(doc, 0.0) + 1.0 / (RRF_K + rank + 1)
    return sorted(score, key=score.get, reverse=True)[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=200)
    ap.add_argument("--model", default="text-embedding-3-small")
    ap.add_argument("--out", default="results/stale_write_time.json")
    args = ap.parse_args()

    xs = stale.load()
    sample = [x for x in xs if x.conflict_type == "T1"][: args.per_type]
    sample += [x for x in xs if x.conflict_type == "T2"][: args.per_type]

    emb = Embedder(args.model)
    queries = [x.new_observation for x in sample]
    todo = emb.missing(queries)
    if todo:
        print(f"embedding {len(todo)} queries")
        emb.add(queries)

    hits = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    rand = collections.defaultdict(list)

    for n, x in enumerate(sample, 1):
        u = past_units(x)
        texts = [t for _, t in u]
        n_gold = sum(1 for s, _ in u if s == x.old_session)

        bm = BM25Okapi([tokenize(t) for t in texts]).get_scores(tokenize(x.new_observation))
        bm_rank = sorted(range(len(u)), key=lambda i: bm[i], reverse=True)

        qv = emb.get([x.new_observation])[0]
        dn = emb.get(texts) @ qv
        dn_rank = sorted(range(len(u)), key=lambda i: dn[i], reverse=True)

        pool = max(KS) * 4
        hy_rank = rrf([bm_rank[:pool], dn_rank[:pool]], max(KS))

        for method, ranking in (("bm25", bm_rank), ("dense", dn_rank), ("hybrid", hy_rank)):
            for k in KS:
                got = float(any(u[i][0] == x.old_session for i in ranking[:k]))
                hits[method][x.conflict_type][k].append(got)
        for k in KS:
            rand[k].append(chance(len(u), n_gold, k))
        if n % 50 == 0:
            print(f"\r  {n}/{len(sample)}", end="", flush=True)

    print()
    table = {}
    for m in METHODS:
        table[m] = []
        for k in KS:
            t1, t2 = hits[m]["T1"][k], hits[m]["T2"][k]
            z, p = ztest(t1, t2)
            table[m].append(
                {
                    "k": k,
                    "T1": round(sum(t1) / len(t1), 4),
                    "T2": round(sum(t2) / len(t2), 4),
                    "gap": round(sum(t1) / len(t1) - sum(t2) / len(t2), 4),
                    "p": round(p, 4),
                    "chance": round(sum(rand[k]) / len(rand[k]), 4),
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "n_per_type": args.per_type, "results": table}, indent=2))

    print(f"\nwrite-time conflict retrieval, n={args.per_type} per type, {args.model}\n")
    print(f"{'method':9s}{'k':>4}{'T1':>8}{'T2':>8}{'gap':>8}{'p':>9}{'chance':>9}")
    print("-" * 55)
    for m in METHODS:
        for r in table[m]:
            print(
                f"{m:9s}{r['k']:>4}{r['T1']:>8.3f}{r['T2']:>8.3f}"
                f"{r['gap']:>8.3f}{r['p']:>9.4f}{r['chance']:>9.3f}"
            )
        print()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
