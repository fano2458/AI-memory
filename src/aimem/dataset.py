import random
from datetime import date, timedelta

from .render import noise_turn, render, render_implicit
from .scenario import generate
from .schema import Instance, Query, Session, Turn
from .world import SUPERSEDE


def to_instance(scenario, seed=0, implicit=False, events_per_session=3, noise_turns=2):
    rng = random.Random(seed)
    target = scenario.queries[0].slot
    chain_ids = set(scenario.queries[0].gold_historical)

    sessions, gold_sessions = [], []
    buf, day = [], 0
    for e in scenario.events:
        text = None
        if implicit and (e.entity, e.relation) == target and e.op == SUPERSEDE:
            text = render_implicit(e, rng)
        text = text or render(e, rng)
        buf.append((e, text))
        if len(buf) >= events_per_session:
            sessions.append(_session(len(sessions), day, buf, chain_ids, rng, noise_turns))
            if any(e.event_id in chain_ids for e, _ in buf):
                gold_sessions.append(len(sessions) - 1)
            buf, day = [], day + rng.randint(1, 5)
    if buf:
        sessions.append(_session(len(sessions), day, buf, chain_ids, rng, noise_turns))
        if any(e.event_id in chain_ids for e, _ in buf):
            gold_sessions.append(len(sessions) - 1)

    q = scenario.queries[0]
    entity, relation = target
    question = f"What is {entity}'s current {relation.replace('_', ' ')}?"
    return Instance(
        instance_id=scenario.scenario_id,
        sessions=sessions,
        queries=[Query(f"{scenario.scenario_id}:q0", "current_state", question)],
        gold_sessions=gold_sessions,
        conflict_type="implicit" if implicit else "explicit",
        answer=q.answer,
        explanation=f"{entity}.{relation} is {q.answer}",
        query_date=sessions[-1].timestamp if sessions else "",
    )


def _session(index, day, buf, chain_ids, rng, noise_turns):
    turns = []
    for e, text in buf:
        turns.append(Turn("user", text, e.event_id in chain_ids))
        turns.append(Turn("assistant", "Noted, thanks for telling me.", False))
    for _ in range(noise_turns):
        turns.append(Turn("user", noise_turn(rng), False))
        turns.append(Turn("assistant", "Happy to help with that.", False))
    stamp = (date(2024, 1, 8) + timedelta(days=day)).strftime("%Y/%m/%d (%a)")
    return Session(index=index, timestamp=stamp, turns=turns)


def build(n=100, chain_len=4, ops=None, n_distractor_slots=8, implicit=False, seed=0):
    out = []
    for i in range(n):
        sc = generate(
            scenario_id=f"gen_{i:05d}",
            seed=seed + i,
            chain_len=chain_len,
            n_distractor_slots=n_distractor_slots,
            ops=ops or [SUPERSEDE],
        )
        out.append(to_instance(sc, seed=seed + i, implicit=implicit))
    return out
