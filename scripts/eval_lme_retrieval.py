import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rank_bm25 import BM25Okapi

from aimem import longmemeval as lme
from aimem.embed import Embedder
from aimem.retrieval import sessions_of, tokenize, units

KS = [1, 2, 4, 8, 16, 32]
METHODS = ["bm25", "dense"]


def summarise(bucket):
    return {
        f"k={k}": {
            "csr": round(sum(a for a, _ in v) / len(v), 4),
            "ach": round(sum(b for _, b in v) / len(v), 4),
            "n": len(v),
        }
        for k, v in sorted(bucket.items())
    }


def show(title, rows, keys):
    print(f"\n== {title} ==")
    header = f"{'group':26s}" + "".join(f"  CSR@{k:<2d} ACH@{k:<2d}" for k in KS)
    print(header)
    print("-" * len(header))
    for name in keys:
        by_k = rows[name]
        line = f"{str(name):26s}"
        for k in KS:
            c = by_k[f"k={k}"]
            line += f"  {c['csr']:.2f}   {c['ach']:.2f} "
        print(line + f"  n={by_k['k=1']['n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="s")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="text-embedding-3-small")
    ap.add_argument("--out", default="results/lme_retrieval.json")
    args = ap.parse_args()

    xs = lme.load(tier=args.tier, limit=args.limit)
    emb = Embedder(args.model)
    qs = [x.queries[0].text for x in xs]
    todo = emb.missing(qs)
    if todo:
        print(f"embedding {len(todo)} questions")
        emb.add(qs)

    by_type = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    by_size = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    overall = collections.defaultdict(lambda: collections.defaultdict(list))

    for n, x in enumerate(xs, 1):
        u = units(x)
        texts = [t for _, t in u]
        gold = set(x.gold_sessions)
        q = x.queries[0].text
        rankings = {
            "bm25": BM25Okapi([tokenize(t) for t in texts]).get_scores(tokenize(q)),
            "dense": emb.get(texts) @ emb.get([q])[0],
        }
        for m, sc in rankings.items():
            ranked = sorted(range(len(u)), key=lambda i: sc[i], reverse=True)
            for k in KS:
                hit = set(sessions_of(x, ranked[:k]))
                pair = (len(hit & gold) / len(gold), float(gold <= hit))
                by_type[m][x.conflict_type][k].append(pair)
                by_size[m][len(gold)][k].append(pair)
                overall[m][k].append(pair)
        if n % 50 == 0:
            print(f"\r  {n}/{len(xs)}", end="", flush=True)
    print()

    out_rows = {
        m: {
            "overall": summarise(overall[m]),
            "by_question_type": {t: summarise(b) for t, b in by_type[m].items()},
            "by_conflict_set_size": {str(s): summarise(b) for s, b in by_size[m].items()},
        }
        for m in METHODS
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tier": args.tier, "model": args.model, "results": out_rows}, indent=2))

    for m in METHODS:
        show(f"{m} by question type", out_rows[m]["by_question_type"],
             sorted(out_rows[m]["by_question_type"], key=lambda t: -out_rows[m]["by_question_type"][t]["k=1"]["n"]))
        show(f"{m} by |C*|", out_rows[m]["by_conflict_set_size"],
             sorted(out_rows[m]["by_conflict_set_size"], key=int))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
