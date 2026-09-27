import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimem.scenario import generate
from aimem.world import SUPERSEDE, TEMPORARY, UNCERTAIN


def test_chain_length_controls_conflict_set():
    for n in (1, 2, 3, 4, 5):
        s = generate(seed=1, chain_len=n, n_distractor_slots=3, ops=[SUPERSEDE])
        q = s.queries[0]
        assert len(q.gold_historical) == n
        assert len(q.gold_conflicting) == n - 1


def test_gold_sets_partition_history():
    s = generate(seed=2, chain_len=4, ops=[SUPERSEDE, TEMPORARY, UNCERTAIN])
    q = s.queries[0]
    assert sorted(q.gold_current + q.gold_conflicting) == sorted(q.gold_historical)
    assert not set(q.gold_current) & set(q.gold_conflicting)


def test_answer_matches_ledger():
    s = generate(seed=3, chain_len=4, ops=[SUPERSEDE])
    q = s.queries[0]
    assert q.answer == s.world.current(q.slot, q.now).value


def test_uncertain_is_never_current():
    s = generate(seed=4, chain_len=5, ops=[UNCERTAIN])
    q = s.queries[0]
    uncertain = {e.event_id for e in s.events if e.op == UNCERTAIN}
    assert not set(q.gold_current) & uncertain


def test_deterministic():
    a = generate(seed=7, chain_len=4, ops=[SUPERSEDE, TEMPORARY])
    b = generate(seed=7, chain_len=4, ops=[SUPERSEDE, TEMPORARY])
    assert [(e.event_id, e.time, e.value, e.op) for e in a.events] == [
        (e.event_id, e.time, e.value, e.op) for e in b.events
    ]


def test_distractors_do_not_touch_target_slot():
    s = generate(seed=5, chain_len=3, n_distractor_slots=8, ops=[SUPERSEDE])
    target = s.queries[0].slot
    ids = {e.event_id for e in s.events if (e.entity, e.relation) == target}
    assert set(s.queries[0].gold_historical) <= ids
