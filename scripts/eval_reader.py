import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimem import longmemeval as lme
from aimem.embed import Embedder
from aimem.reader import Reader
from aimem.retrieval import units

READ_SYS = (
    "You answer questions about a user from excerpts of their past conversations. "
    "Excerpts are dated. Use only the excerpts. Be concise. "
    "If the answer is not in the excerpts, say you do not know."
)
JUDGE_SYS = (
    "You grade a model answer to a question about a user conversation history.\n"
    "It is CORRECT if it agrees with the reference on the key information, even if phrased "
    "differently, more verbosely, or with extra detail.\n"
    "If the reference says the information is insufficient or unknown, an answer that declines "
    "to answer is CORRECT.\n"
    "It is INCORRECT if it states a different fact, number, or entity than the reference.\n"
    "Reply with exactly one word: yes or no."
)


def oracle_evidence(x):
    return [(x.sessions[g].timestamp, x.sessions[g].text) for g in sorted(x.gold_sessions)]


def gold_evidence(x):
    ev = [
        (s.timestamp, t.content)
        for s in x.sessions
        if s.index in x.gold_sessions
        for t in s.turns
        if t.has_answer
    ]
    return ev or oracle_evidence(x)


def dense_evidence(x, emb, k):
    u = units(x)
    texts = [t for _, t in u]
    scores = emb.get(texts) @ emb.get([x.queries[0].text])[0]
    top = sorted(range(len(u)), key=lambda i: scores[i], reverse=True)[:k]
    return [(x.sessions[u[i][0]].timestamp, texts[i]) for i in sorted(top)]


def build_prompt(x, evidence):
    blocks = "\n\n".join(f"[{ts}]\n{txt}" for ts, txt in evidence)
    return f"Today is {x.query_date}.\n\nExcerpts:\n{blocks}\n\nQuestion: {x.queries[0].text}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--judge", default="gpt-5.4-mini")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", default="oracle,dense8")
    ap.add_argument("--out", default="results/lme_reader.json")
    args = ap.parse_args()

    xs = lme.load("s")
    if args.limit:
        random.Random(args.seed).shuffle(xs)
        xs = xs[: args.limit]
    emb = Embedder()
    reader = Reader(args.model)
    judge = Reader(args.judge, max_tokens=8)

    conditions = args.conditions.split(",")
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    cost = collections.defaultdict(lambda: [0, 0])

    for n, x in enumerate(xs, 1):
        for cond in conditions:
            if cond == "gold":
                ev = gold_evidence(x)
            elif cond == "oracle":
                ev = oracle_evidence(x)
            else:
                ev = dense_evidence(x, emb, int(cond[5:]))
            out = reader.ask(READ_SYS, build_prompt(x, ev))
            verdict = judge.ask(
                JUDGE_SYS,
                f"Question: {x.queries[0].text}\nReference answer: {x.answer}\n"
                f"Model answer: {out['text']}",
            )
            ok = verdict["text"].strip().lower().startswith("yes")
            acc[cond]["overall"].append(ok)
            acc[cond][f"type:{x.conflict_type}"].append(ok)
            acc[cond][f"cstar:{len(x.gold_sessions)}"].append(ok)
            cost[cond][0] += out["in_tokens"]
            cost[cond][1] += out["out_tokens"]
        if n % 25 == 0:
            print(f"\r  {n}/{len(xs)}", end="", flush=True)
    print()

    rows = {
        c: {k: {"acc": round(sum(v) / len(v), 4), "n": len(v)} for k, v in sorted(b.items())}
        for c, b in acc.items()
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "reader": args.model,
                "judge": args.judge,
                "n": len(xs),
                "results": rows,
                "reader_tokens": {c: {"in": v[0], "out": v[1]} for c, v in cost.items()},
            },
            indent=2,
        )
    )

    groups = sorted({k for b in rows.values() for k in b})
    print(f"\n{'group':30s}" + "".join(f"{c:>12s}" for c in conditions))
    print("-" * (30 + 12 * len(conditions)))
    for g in ["overall"] + [x for x in groups if x != "overall"]:
        line = f"{g:30s}"
        for c in conditions:
            r = rows[c].get(g)
            line += f"{r['acc']:>12.3f}" if r else f"{'-':>12s}"
        n = rows[conditions[0]].get(g, {}).get("n", "")
        print(line + f"   n={n}")
    print()
    for c in conditions:
        print(f"{c}: {cost[c][0]:,} in / {cost[c][1]:,} out reader tokens")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
