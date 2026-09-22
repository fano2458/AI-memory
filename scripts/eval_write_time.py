import argparse
import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rank_bm25 import BM25Okapi

from aimem import stale
from aimem.retrieval import tokenize

KS = [1, 2, 4, 8, 16, 32]


def past_units(inst):
    return [
        (s.index, f"{t.role}: {t.content}")
        for s in inst.sessions
        if s.index < inst.new_session
        for t in s.turns
    ]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=200)
    ap.add_argument("--out", default="results/stale_write_time_bm25.json")
    args = ap.parse_args()

    xs = stale.load()
    sample = [x for x in xs if x.conflict_type == "T1"][: args.per_type]
    sample += [x for x in xs if x.conflict_type == "T2"][: args.per_type]

    hits = collections.defaultdict(lambda: collections.defaultdict(list))
    rand = collections.defaultdict(list)

    for x in sample:
        u = past_units(x)
        n_gold = sum(1 for s, _ in u if s == x.old_session)
        scores = BM25Okapi([tokenize(t) for _, t in u]).get_scores(tokenize(x.new_observation))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        for k in KS:
            hits[x.conflict_type][k].append(
                float(any(u[i][0] == x.old_session for i in ranked[:k]))
            )
            rand[k].append(chance(len(u), n_gold, k))

    table = []
    for k in KS:
        t1, t2 = hits["T1"][k], hits["T2"][k]
        z, p = ztest(t1, t2)
        table.append(
            {
                "k": k,
                "T1": round(sum(t1) / len(t1), 4),
                "T2": round(sum(t2) / len(t2), 4),
                "gap": round(sum(t1) / len(t1) - sum(t2) / len(t2), 4),
                "z": round(z, 2),
                "p": round(p, 4),
                "chance": round(sum(rand[k]) / len(rand[k]), 4),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_per_type": len(hits["T1"][1]), "hit_at_k": table}, indent=2))

    print(f"write-time conflict retrieval (BM25), n={len(hits['T1'][1])} per type\n")
    print(f"{'k':>4} {'T1':>7} {'T2':>7} {'gap':>7} {'z':>6} {'p':>8} {'chance':>8}")
    print("-" * 52)
    for r in table:
        print(
            f"{r['k']:>4} {r['T1']:>7.3f} {r['T2']:>7.3f} {r['gap']:>7.3f} "
            f"{r['z']:>6.2f} {r['p']:>8.4f} {r['chance']:>8.3f}"
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
