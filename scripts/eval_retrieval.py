"""Session-level conflict-set retrieval on STALE. No LLM required."""

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rank_bm25 import BM25Okapi

from aimem import stale
from aimem.retrieval import sessions_of, tokenize, units

KS = [1, 2, 4, 8, 16, 32]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=60)
    ap.add_argument("--out", default="results/stale_retrieval_bm25.json")
    args = ap.parse_args()

    xs = stale.load()
    t1 = [x for x in xs if x.conflict_type == "T1"][: args.per_type]
    t2 = [x for x in xs if x.conflict_type == "T2"][: args.per_type]
    sample = t1 + t2

    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in sample:
        bm = BM25Okapi([tokenize(t) for _, t in units(x)])
        gold = set(x.gold_sessions)
        for q in x.queries:
            scores = bm.get_scores(tokenize(q.text))
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for k in KS:
                hit = set(sessions_of(x, ranked[:k]))
                agg[(x.conflict_type, q.dimension)][k].append(
                    (len(hit & gold) / len(gold), float(gold <= hit))
                )

    rows = {}
    for (ctype, dim), by_k in sorted(agg.items()):
        rows[f"{ctype}/{dim}"] = {
            f"k={k}": {
                "csr": round(sum(a for a, _ in v) / len(v), 4),
                "ach": round(sum(b for _, b in v) / len(v), 4),
                "n": len(v),
            }
            for k, v in sorted(by_k.items())
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_instances": len(sample), "results": rows}, indent=2))

    header = "type/dim   " + "".join(f"  CSR@{k:<2d} ACH@{k:<2d}" for k in KS)
    print(header)
    print("-" * len(header))
    for name, by_k in rows.items():
        line = f"{name:11s}"
        for k in KS:
            c = by_k[f"k={k}"]
            line += f"  {c['csr']:.2f}   {c['ach']:.2f} "
        print(line)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
