import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from aimem.embed import Embedder
from aimem.render import render
from aimem.scenario import DOMAINS
from aimem.world import SET, SUPERSEDE, Event, World

KS = [1, 2, 4, 8, 16]


def build(n_slots, redundancy, seed):
    rng = random.Random(seed)
    rels = [(r, v) for d in DOMAINS.values() for r, v in d.items()]
    world = World()
    events, slots = [], []
    for i in range(n_slots):
        relation, values = rng.choice(rels)
        slots.append((f"person_{i:04d}", relation, values))
    plan = [i for i in range(n_slots) for _ in range(redundancy)]
    rng.shuffle(plan)
    cursor = collections.Counter()
    t = 0
    for eid, i in enumerate(plan, 1):
        entity, relation, values = slots[i]
        j = cursor[i]
        cursor[i] += 1
        t += 1
        ev = Event(eid, t, entity, relation, values[j % len(values)], SET if j == 0 else SUPERSEDE)
        events.append(ev)
        world.apply(ev)
    return world, events, slots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=400)
    ap.add_argument("--queries", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/active_set.json")
    args = ap.parse_args()

    emb = Embedder()
    rows = {}
    for redundancy in (1, 2, 4, 8):
        world, events, slots = build(args.slots, redundancy, args.seed)
        rng = random.Random(args.seed)

        texts = [render(e, random.Random(e.event_id)) for e in events]
        active_ids = {
            world.current((e, r), 10**9).event_id
            for e, r, _ in slots
            if world.current((e, r), 10**9)
        }
        active_pos = [i for i, e in enumerate(events) if e.event_id in active_ids]

        picks = rng.sample(range(len(slots)), min(args.queries, len(slots)))
        queries = {
            i: f"What is the current {slots[i][1].replace('_', ' ')} for {slots[i][0]}?"
            for i in picks
        }
        todo = emb.missing(texts + list(queries.values()))
        if todo:
            print(f"redundancy {redundancy}: embedding {len(todo)} new strings")
            emb.add(texts + list(queries.values()))
        mat = emb.get(texts)
        res = collections.defaultdict(lambda: collections.defaultdict(list))
        for i in picks:
            entity, relation, _ = slots[i]
            cur = world.current((entity, relation), 10**9)
            if cur is None:
                continue
            query = queries[i]
            qv = emb.get([query])[0]
            gold_pos = next(p for p, e in enumerate(events) if e.event_id == cur.event_id)
            for name, pool in (("full_log", list(range(len(events)))), ("active", active_pos)):
                idx = np.array(pool)
                order = idx[np.argsort(-(mat[idx] @ qv))]
                for k in KS:
                    res[name][k].append(float(gold_pos in set(order[:k].tolist())))

        rows[str(redundancy)] = {
            "events": len(events),
            "active": len(active_pos),
            "hit": {
                name: {f"k={k}": round(sum(v) / len(v), 4) for k, v in sorted(b.items())}
                for name, b in res.items()
            },
        }
        print(f"redundancy {redundancy}: {len(events)} events -> {len(active_pos)} active")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))

    print(f"\n{'events/slot':>12}{'log':>8}{'active':>8}  " + "".join(f"{'full@'+str(k):>10}" for k in KS) + "  " + "".join(f"{'act@'+str(k):>10}" for k in KS))
    print("-" * (28 + 20 * len(KS)))
    for r, v in rows.items():
        line = f"{r:>12}{v['events']:>8,}{v['active']:>8,}  "
        line += "".join(f"{v['hit']['full_log'][f'k={k}']:>10.3f}" for k in KS) + "  "
        line += "".join(f"{v['hit']['active'][f'k={k}']:>10.3f}" for k in KS)
        print(line)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
