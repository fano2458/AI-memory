"""Audit how many competing values each state slot has, per benchmark.

Two different quantities are reported and must not be confused:
  evidence set   = how many memories are needed to answer
  conflict chain = how many values one slot has held over time
"""

import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimem import longmemeval as lme
from aimem import stale


def report(name, counter, note=""):
    total = sum(counter.values())
    print(f"\n{name}  (n={total})")
    for k in sorted(counter):
        print(f"   chain length {k}: {counter[k]:>5}   {counter[k] / total * 100:5.1f}%")
    if note:
        print(f"   {note}")


def audit_stale():
    xs = stale.load()
    c = collections.Counter(len(x.gold_sessions) for x in xs)
    report("STALE", c, "gold = [session with M_old, session with M_new]")


def audit_lme():
    xs = lme.load("s")
    ev = collections.Counter(len(x.gold_sessions) for x in xs)
    print("\nLongMemEval-s EVIDENCE set size (all types)")
    for k in sorted(ev):
        print(f"   {k}: {ev[k]:>5}")
    ku = collections.Counter(
        len(x.gold_sessions) for x in xs if x.conflict_type == "knowledge-update"
    )
    report("LongMemEval-s knowledge-update CONFLICT chain", ku)
    big = [x for x in xs if len(x.gold_sessions) > 2]
    types = collections.Counter(x.conflict_type for x in big)
    print(f"   evidence>2 comes from: {dict(types)}  (aggregation, not supersession)")


def audit_mab():
    import pyarrow.parquet as pq

    p = Path("data/external/mab/conflict.parquet")
    if not p.exists():
        print("\nMemoryAgentBench: not downloaded")
        return
    rows = pq.read_table(p).to_pylist()
    facts = [
        line.split(". ", 1)[1]
        for line in rows[0]["context"].split("\n")
        if re.match(r"^\d+\. ", line)
    ]
    # a fact is "<stem> <verb> <value>"; group by stem+verb to find repeated slots
    pat = re.compile(
        r"^(.*?)\s+(was born in the city of|is located in the continent of|plays the position of"
        r"|is married to|was founded by|is the country of citizenship of|died in the city of"
        r"|worked in the city of|is|was)\s+(.*?)\.?$"
    )
    slots = collections.defaultdict(set)
    unparsed = 0
    for f in facts:
        m = pat.match(f)
        if not m:
            unparsed += 1
            continue
        slots[(m.group(1), m.group(2))].add(m.group(3))
    c = collections.Counter(len(v) for v in slots.values())
    report(
        "MemoryAgentBench Conflict_Resolution",
        c,
        f"distinct slots={len(slots)}, unparsed lines={unparsed}",
    )




UNCERTAIN_RE = re.compile(
    r"\b(might|may |maybe|possibly|considering|thinking (about|of)|planning to|hoping to"
    r"|if I |could be|not sure|undecided|potentially|leaning towards?|weighing)\b",
    re.I,
)
TEMPORARY_RE = re.compile(
    r"\b(for (a few|a couple|two|three|several) (days|weeks|nights|months)|temporarily"
    r"|for now|for the time being|until (the|next|then)|while I'm|just for|short stay"
    r"|staying (with|at) .{0,20} (for|until)|on loan|house.?sitting|sublet)\b",
    re.I,
)


def scan(name, texts):
    u = [t for t in texts if UNCERTAIN_RE.search(t)]
    t_ = [t for t in texts if TEMPORARY_RE.search(t)]
    print(f"\n{name}  (n={len(texts)})")
    print(f"   uncertainty markers : {len(u):>5}  ({len(u) / max(len(texts), 1) * 100:.1f}%)")
    print(f"   temporary markers   : {len(t_):>5}  ({len(t_) / max(len(texts), 1) * 100:.1f}%)")
    for label, hits in (("uncertain", u), ("temporary", t_)):
        for h in hits[:2]:
            print(f"     [{label}] {h[:130]}")


def audit_semantics():
    xs = stale.load()
    scan("STALE M_new (the update itself)", [x.new_observation for x in xs])

    lx = lme.load("s")
    ku = [x for x in lx if x.conflict_type == "knowledge-update"]
    gold_turns = [t.content for x in ku for s in x.sessions if s.index in x.gold_sessions
                  for t in s.turns if t.has_answer]
    scan("LongMemEval knowledge-update gold evidence turns", gold_turns)

    import pyarrow.parquet as pq
    p = Path("data/external/mab/conflict.parquet")
    if p.exists():
        rows = pq.read_table(p).to_pylist()
        facts = [line.split(". ", 1)[1] for line in rows[0]["context"].split("\n")
                 if re.match(r"^\d+\. ", line)]
        scan("MemoryAgentBench facts", facts)


if __name__ == "__main__":
    audit_stale()
    audit_lme()
    audit_mab()
    audit_semantics()
