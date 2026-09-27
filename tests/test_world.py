import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimem.world import CANCEL, REFINE, REVERT, SET, SUPERSEDE, TEMPORARY, UNCERTAIN, Event, World

SLOT = ("user", "location")


def build(*specs):
    w = World()
    for i, (t, op, val, *rest) in enumerate(specs):
        w.apply(Event(i, t, "user", "location", val, op, until=rest[0] if rest else -1))
    return w


def test_set_then_supersede():
    w = build((1, SET, "Seattle"), (5, SUPERSEDE, "Austin"))
    assert w.current(SLOT, 10).value == "Austin"
    assert w.current(SLOT, 3).value == "Seattle"
    old = w.history(SLOT)[0]
    assert old.status == "SUPERSEDED" and old.valid_to == 5


def test_chain_of_three():
    w = build((1, SET, "v1"), (5, SUPERSEDE, "v2"), (9, SUPERSEDE, "v3"))
    assert w.current(SLOT, 10).value == "v3"
    assert [f.value for f in w.history(SLOT)] == ["v1", "v2", "v3"]
    assert sum(f.open for f in w.history(SLOT)) == 1


def test_temporary_does_not_erase_permanent():
    w = build((1, SET, "Astana"), (5, TEMPORARY, "Almaty", 8))
    assert w.current(SLOT, 6).value == "Almaty"
    assert w.current(SLOT, 9).value == "Astana"


def test_revert():
    w = build((1, SET, "v1"), (5, SUPERSEDE, "v2"), (9, REVERT, "v1"))
    assert w.current(SLOT, 10).value == "v1"
    assert len(w.history(SLOT)) == 3


def test_uncertain_does_not_overwrite():
    w = build((1, SET, "Almaty"), (5, UNCERTAIN, "Astana"))
    assert w.current(SLOT, 10).value == "Almaty"
    assert any(f.status == "UNCERTAIN" for f in w.history(SLOT))


def test_cancel_leaves_no_current():
    w = build((1, SET, "Seattle"), (5, CANCEL, "Seattle"))
    assert w.current(SLOT, 10) is None


def test_refine():
    w = build((1, SET, "Europe"), (5, REFINE, "France"))
    assert w.current(SLOT, 10).value == "France"
    assert w.history(SLOT)[0].status == "REFINED"


def test_current_is_none_before_first_event():
    w = build((5, SET, "Seattle"))
    assert w.current(SLOT, 1) is None
