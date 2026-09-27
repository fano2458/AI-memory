import argparse
import collections
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from aimem import longmemeval as lme
from aimem.embed import Embedder

KS = [4, 8, 16, 32]
SIZES = [500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 246750]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/lme_scaling.json")
    args = ap.parse_args()

    xs = lme.load("s")
    emb = Embedder()

    pool_text, owner, is_gold = [], [], []
    for x in xs:
        for s in x.sessions:
            for t in s.turns:
                pool_text.append(t.content)
                owner.append(x.instance_id)
                is_gold.append(t.has_answer)
    print(f"pool: {len(pool_text):,} turns")

    mat = emb.get(pool_text)
    by_owner = collections.defaultdict(list)
    for i, o in enumerate(owner):
        by_owner[o].append(i)

    rng = random.Random(args.seed)
    sample = [x for x in xs if any(is_gold[i] for i in by_owner[x.instance_id])]
    rng.shuffle(sample)
    sample = sample[: args.questions]

    owner_arr = np.array(owner)
    nprng = np.random.default_rng(args.seed)

    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for n, x in enumerate(sample, 1):
        own = np.array(by_owner[x.instance_id])
        gold = {i for i in by_owner[x.instance_id] if is_gold[i]}
        others = np.flatnonzero(owner_arr != x.instance_id)
        scores = mat @ emb.get([x.queries[0].text])[0]

        for size in SIZES:
            extra = max(0, size - len(own))
            picked = nprng.choice(others, min(extra, len(others)), replace=False)
            idx = np.concatenate([own, picked])
            order = idx[np.argsort(-scores[idx])]
            for k in KS:
                top = set(order[:k].tolist())
                res[size][k].append((len(top & gold) / len(gold), float(gold <= top)))
        if n % 25 == 0:
            print(f"\r  {n}/{len(sample)}", end="", flush=True)
    print()

    rows = {
        str(size): {
            f"k={k}": {
                "recall": round(st.mean(a for a, _ in v), 4),
                "complete": round(st.mean(b for _, b in v), 4),
            }
            for k, v in sorted(by_k.items())
        }
        for size, by_k in sorted(res.items())
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_questions": len(sample), "results": rows}, indent=2))

    print(f"\n{'store turns':>12}" + "".join(f"  rec@{k:<3d} all@{k:<3d}" for k in KS))
    print("-" * (12 + 16 * len(KS)))
    for size in SIZES:
        line = f"{size:>12,}"
        for k in KS:
            c = rows[str(size)][f"k={k}"]
            line += f"   {c['recall']:.3f}   {c['complete']:.3f}"
        print(line)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
