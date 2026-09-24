import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rank_bm25 import BM25Okapi

from aimem import stale
from aimem.embed import Embedder
from aimem.retrieval import sessions_of, tokenize, units

KS = [1, 2, 4, 8, 16, 32]
METHODS = ["bm25", "dense"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=100)
    ap.add_argument("--model", default="text-embedding-3-small")
    ap.add_argument("--out", default="results/stale_read_time.json")
    args = ap.parse_args()

    xs = stale.load()
    sample = [x for x in xs if x.conflict_type == "T1"][: args.per_type]
    sample += [x for x in xs if x.conflict_type == "T2"][: args.per_type]

    emb = Embedder(args.model)
    qs = [q.text for x in sample for q in x.queries]
    todo = emb.missing(qs)
    if todo:
        print(f"embedding {len(todo)} queries")
        emb.add(qs)

    agg = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))

    for n, x in enumerate(sample, 1):
        u = units(x)
        texts = [t for _, t in u]
        gold = set(x.gold_sessions)
        bm = BM25Okapi([tokenize(t) for t in texts])
        mat = emb.get(texts)
        for q in x.queries:
            scores = {
                "bm25": bm.get_scores(tokenize(q.text)),
                "dense": mat @ emb.get([q.text])[0],
            }
            for m, sc in scores.items():
                ranked = sorted(range(len(u)), key=lambda i: sc[i], reverse=True)
                for k in KS:
                    hit = set(sessions_of(x, ranked[:k]))
                    agg[m][(x.conflict_type, q.dimension)][k].append(
                        (len(hit & gold) / len(gold), float(gold <= hit))
                    )
        if n % 50 == 0:
            print(f"\r  {n}/{len(sample)}", end="", flush=True)
    print()

    out_rows = {}
    for m in METHODS:
        out_rows[m] = {
            f"{ct}/{dim}": {
                f"k={k}": {
                    "csr": round(sum(a for a, _ in v) / len(v), 4),
                    "ach": round(sum(b for _, b in v) / len(v), 4),
                }
                for k, v in sorted(by_k.items())
            }
            for (ct, dim), by_k in sorted(agg[m].items())
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "results": out_rows}, indent=2))

    for m in METHODS:
        print(f"\n== {m} ==")
        header = "type/dim   " + "".join(f"  CSR@{k:<2d} ACH@{k:<2d}" for k in KS)
        print(header)
        print("-" * len(header))
        for name, by_k in out_rows[m].items():
            line = f"{name:11s}"
            for k in KS:
                c = by_k[f"k={k}"]
                line += f"  {c['csr']:.2f}   {c['ach']:.2f} "
            print(line)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
