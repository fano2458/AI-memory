import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rank_bm25 import BM25Okapi

from aimem.dataset import build
from aimem.embed import Embedder
from aimem.reader import Reader
from aimem.retrieval import tokenize, units
from aimem.world import REVERT, SUPERSEDE, TEMPORARY, UNCERTAIN

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_reader import JUDGE_SYS, READ_SYS, build_prompt

MIXES = {
    "supersede": [SUPERSEDE],
    "+revert": [SUPERSEDE, REVERT],
    "+temporary": [SUPERSEDE, TEMPORARY],
    "+uncertain": [SUPERSEDE, UNCERTAIN],
    "all mixed": [SUPERSEDE, TEMPORARY, UNCERTAIN, REVERT],
}
K = 8


def gold_turns(x):
    return {i for i, t in enumerate(t for s in x.sessions for t in s.turns) if t.has_answer}


def evidence(x, ids):
    flat = [(s.timestamp, t) for s in x.sessions for t in s.turns]
    return [(ts, t.content) for i, (ts, t) in enumerate(flat) if i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--chain-len", type=int, default=4)
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--no-reader", action="store_true")
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--out", default="results/generated_baselines.json")
    args = ap.parse_args()

    emb = Embedder()
    reader = Reader(args.model)
    judge = Reader(args.model, max_tokens=256)
    rows = {}

    for name, ops in MIXES.items():
        xs = build(n=args.n, chain_len=args.chain_len, ops=ops, seed=0)
        texts = [t.content for x in xs for s in x.sessions for t in s.turns]
        qs = [x.queries[0].text for x in xs]
        if not args.no_dense:
            todo = emb.missing(texts + qs)
            if todo:
                emb.add(texts + qs)

        acc = collections.defaultdict(list)
        for x in xs:
            u = units(x)
            pool = [t for _, t in u]
            gold = gold_turns(x)
            flat = [t for s in x.sessions for t in s.turns]

            # deterministic: latest gold-chain turn wins
            latest = max(gold)
            acc["determ"].append(flat[latest].content.find(x.answer) >= 0)

            q = x.queries[0].text
            bm = BM25Okapi([tokenize(t) for t in pool]).get_scores(tokenize(q))
            scorers = [("bm25", bm)]
            if not args.no_dense:
                scorers.append(("dense", emb.get(pool) @ emb.get([q])[0]))
            for label, sc in scorers:
                rank = sorted(range(len(pool)), key=lambda i: sc[i], reverse=True)[:K]
                acc[f"{label}_complete"].append(float(gold <= set(rank)))
                if label == "dense" and not args.no_reader:
                    out = reader.ask(READ_SYS, build_prompt(x, evidence(x, set(rank))))
                    v = judge.ask(
                        JUDGE_SYS,
                        f"Question: {q}\nReference answer: {x.answer}\nModel answer: {out['text']}",
                    )
                    acc["reader_dense"].append("yes" in v["text"].strip().lower()[-8:])

            if not args.no_reader:
                out = reader.ask(READ_SYS, build_prompt(x, evidence(x, gold)))
                v = judge.ask(
                    JUDGE_SYS,
                    f"Question: {q}\nReference answer: {x.answer}\nModel answer: {out['text']}",
                )
                acc["reader_gold"].append("yes" in v["text"].strip().lower()[-8:])

        rows[name] = {k: round(sum(v) / len(v), 3) for k, v in acc.items()}
        print(f"  {name:12s} {rows[name]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"n": args.n, "chain_len": args.chain_len, "k": K, "results": rows}, indent=2))

    cols = ["determ", "bm25_complete", "dense_complete", "reader_gold", "reader_dense"]
    print(f"\n{'op mix':13s}" + "".join(f"{c:>16}" for c in cols))
    print("-" * (13 + 16 * len(cols)))
    for name, r in rows.items():
        print(f"{name:13s}" + "".join(f"{r.get(c, float('nan')):>16.3f}" for c in cols))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
