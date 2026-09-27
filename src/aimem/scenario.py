import random
from dataclasses import dataclass, field

from .world import REFINE, REVERT, SET, SUPERSEDE, TEMPORARY, UNCERTAIN, Event, World

DOMAINS = {
    "personal": {
        "location": ["Seattle", "Austin", "Almaty", "Astana", "Lisbon", "Toronto", "Osaka"],
        "job_title": ["analyst", "team lead", "consultant", "researcher", "product manager"],
        "commute": ["bicycle", "subway", "car", "walking", "tram"],
    },
    "scheduling": {
        "deadline": ["Friday", "Monday", "the 14th", "end of month", "Wednesday"],
        "venue": ["room 3B", "the annex", "the main hall", "a cafe", "online"],
    },
    "software": {
        "api_version": ["v1", "v2", "v3", "v4"],
        "database": ["Postgres", "SQLite", "MySQL", "DuckDB"],
        "deploy_target": ["staging", "production", "the test cluster"],
    },
    "project": {
        "dataset": ["LoCoMo", "the internal logs", "a synthetic set", "the 2024 dump"],
        "model": ["a 7B baseline", "a fine-tuned encoder", "an off-the-shelf reranker"],
    },
    "travel": {
        "destination": ["Porto", "Tbilisi", "Bergen", "Kyoto", "Split"],
        "hotel": ["the riverside inn", "a hostel", "an apartment", "the airport hotel"],
    },
}

CHAIN_OPS = [SUPERSEDE, TEMPORARY, REVERT, REFINE, UNCERTAIN]


@dataclass
class Query:
    slot: tuple[str, str]
    now: int
    answer: str
    gold_current: list[int]
    gold_conflicting: list[int]
    gold_historical: list[int]
    kind: str = "current_state"


@dataclass
class Scenario:
    scenario_id: str
    events: list[Event]
    world: World
    queries: list[Query] = field(default_factory=list)
    target_slots: list[tuple[str, str]] = field(default_factory=list)


def _slots(rng, n):
    pool = [(d, r, vals) for d, rels in DOMAINS.items() for r, vals in rels.items()]
    return rng.sample(pool, min(n, len(pool)))


def generate(
    scenario_id="s0",
    seed=0,
    n_target_slots=1,
    chain_len=3,
    n_distractor_slots=6,
    ops=None,
    entities=("user", "alice", "bob"),
):
    rng = random.Random(seed)
    ops = ops or [SUPERSEDE]
    world = World()
    events: list[Event] = []
    t = 0
    eid = 0
    targets = []

    chosen = _slots(rng, n_target_slots + n_distractor_slots)
    pending = []
    for i, (_domain, relation, values) in enumerate(chosen):
        entity = entities[0] if i < n_target_slots else rng.choice(entities)
        is_target = i < n_target_slots
        vals = rng.sample(values, min(chain_len, len(values)))
        length = chain_len if is_target else 1
        steps = []
        for j in range(length):
            op = SET if j == 0 else rng.choice(ops)
            value = vals[j % len(vals)] if op != REVERT else vals[0]
            steps.append((entity, relation, value, op))
        pending.append(steps)
        if is_target:
            targets.append((entity, relation))

    order = [i for i, steps in enumerate(pending) for _ in steps]
    rng.shuffle(order)
    cursor = [0] * len(pending)
    for i in order:
        entity, relation, value, op = pending[i][cursor[i]]
        cursor[i] += 1
        t += rng.randint(1, 4)
        eid += 1
        ev = Event(
            eid,
            t,
            entity,
            relation,
            value,
            op,
            until=t + rng.randint(1, 3) if op == TEMPORARY else -1,
            confirmed=op != UNCERTAIN,
        )
        events.append(ev)
        world.apply(ev)

    now = t + 1
    queries = [_query(world, slot, now) for slot in targets]
    return Scenario(scenario_id, events, world, queries, targets)


def _query(world, slot, now):
    chain = world.history(slot)
    cur = world.current(slot, now)
    cur_ids = [cur.event_id] if cur else []
    return Query(
        slot=slot,
        now=now,
        answer=cur.value if cur else "unknown",
        gold_current=cur_ids,
        gold_conflicting=[f.event_id for f in chain if f.event_id not in cur_ids],
        gold_historical=[f.event_id for f in chain],
    )
