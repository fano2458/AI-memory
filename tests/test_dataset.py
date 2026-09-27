import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimem.dataset import build, to_instance
from aimem.scenario import generate
from aimem.world import SUPERSEDE, TEMPORARY, UNCERTAIN

MIX = [SUPERSEDE, TEMPORARY, UNCERTAIN]


def test_answer_matches_ledger():
    for seed in range(20):
        sc = generate(seed=seed, chain_len=4, ops=MIX)
        inst = to_instance(sc, seed=seed)
        assert inst.answer == sc.world.current(sc.queries[0].slot, sc.queries[0].now).value


def test_gold_turns_are_exactly_the_chain():
    sc = generate(seed=1, chain_len=4, ops=[SUPERSEDE])
    inst = to_instance(sc, seed=1)
    flagged = sum(1 for s in inst.sessions for t in s.turns if t.has_answer)
    assert flagged == len(sc.queries[0].gold_historical)


def test_gold_sessions_cover_all_gold_turns():
    for seed in range(10):
        sc = generate(seed=seed, chain_len=3, ops=MIX)
        inst = to_instance(sc, seed=seed)
        holding = {s.index for s in inst.sessions if any(t.has_answer for t in s.turns)}
        assert holding == set(inst.gold_sessions)


def test_chain_length_controls_gold_turn_count():
    for n in (2, 3, 4, 5):
        inst = build(n=1, chain_len=n, ops=[SUPERSEDE], seed=3)[0]
        assert sum(1 for s in inst.sessions for t in s.turns if t.has_answer) == n


def test_build_is_deterministic():
    a = build(n=3, chain_len=3, ops=MIX, seed=5)
    b = build(n=3, chain_len=3, ops=MIX, seed=5)
    assert [x.answer for x in a] == [x.answer for x in b]
    assert [t.content for x in a for s in x.sessions for t in s.turns] == [
        t.content for x in b for s in x.sessions for t in s.turns
    ]


def test_instances_have_unique_ids():
    xs = build(n=25, chain_len=3, ops=[SUPERSEDE], seed=0)
    assert len({x.instance_id for x in xs}) == 25
