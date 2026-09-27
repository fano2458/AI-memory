from dataclasses import dataclass, field

SET = "SET"
SUPERSEDE = "SUPERSEDE"
TEMPORARY = "TEMPORARY"
REVERT = "REVERT"
REFINE = "REFINE"
CANCEL = "CANCEL"
UNCERTAIN = "UNCERTAIN"
NOISE = "NOISE"

OPS = [SET, SUPERSEDE, TEMPORARY, REVERT, REFINE, CANCEL, UNCERTAIN, NOISE]


@dataclass
class Event:
    event_id: int
    time: int
    entity: str
    relation: str
    value: str
    op: str
    until: int = -1
    confirmed: bool = True

    @property
    def slot(self):
        return (self.entity, self.relation)


@dataclass
class Fact:
    event_id: int
    value: str
    valid_from: int
    valid_to: int = -1
    status: str = "ACTIVE"
    confirmed: bool = True

    @property
    def open(self):
        return self.valid_to == -1


@dataclass
class World:
    facts: dict[tuple[str, str], list[Fact]] = field(default_factory=dict)

    def apply(self, e: Event):
        chain = self.facts.setdefault(e.slot, [])
        if e.op == NOISE:
            return
        if e.op == UNCERTAIN:
            chain.append(Fact(e.event_id, e.value, e.time, e.time, "UNCERTAIN", False))
            return
        if e.op == CANCEL:
            for f in chain:
                if f.open:
                    f.valid_to = e.time
                    f.status = "CANCELLED"
            return
        if e.op == REVERT:
            self._close_open(chain, e.time, "SUPERSEDED")
            chain.append(Fact(e.event_id, e.value, e.time))
            return
        if e.op == TEMPORARY:
            for f in chain:
                if f.open:
                    f.status = "SUSPENDED"
            chain.append(Fact(e.event_id, e.value, e.time, e.until, "TEMPORARY"))
            return
        if e.op == REFINE:
            self._close_open(chain, e.time, "REFINED")
            chain.append(Fact(e.event_id, e.value, e.time))
            return
        self._close_open(chain, e.time, "SUPERSEDED")
        chain.append(Fact(e.event_id, e.value, e.time))

    @staticmethod
    def _close_open(chain, t, status):
        for f in chain:
            if f.open and f.status != "UNCERTAIN":
                f.valid_to = t
                f.status = status

    def current(self, slot, now):
        live = [
            f
            for f in self.facts.get(slot, [])
            if f.confirmed
            and f.status not in ("CANCELLED", "UNCERTAIN")
            and f.valid_from <= now
            and (f.open or f.valid_to > now)
        ]
        if not live:
            return None
        temp = [f for f in live if f.status == "TEMPORARY"]
        return (temp or live)[-1]

    def history(self, slot):
        return list(self.facts.get(slot, []))
