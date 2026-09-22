import orjson
from pathlib import Path

from .schema import Instance, Query, Session, Turn

DATA = Path("data/external/stale/T1_T2_400_FULL.json")
DIMENSIONS = ["dim1", "dim2", "dim3"]


def load(path: Path = DATA, limit: int | None = None) -> list[Instance]:
    raw = orjson.loads(path.read_bytes())
    if limit:
        raw = raw[:limit]
    return [_parse(r) for r in raw]


def _parse(r: dict) -> Instance:
    sessions = [
        Session(
            index=i,
            timestamp=r["timestamps"][i],
            turns=[Turn(t["role"], t["content"]) for t in turns],
        )
        for i, turns in enumerate(r["haystack_session"])
    ]
    queries = [
        Query(query_id=f"{r['uid']}:{d}", dimension=d, text=r["probing_queries"][f"{d}_query"])
        for d in DIMENSIONS
    ]
    old_session, new_session = r["relevant_session_index"]
    return Instance(
        instance_id=r["uid"],
        sessions=sessions,
        queries=queries,
        gold_sessions=[old_session, new_session],
        old_session=old_session,
        new_session=new_session,
        conflict_type=r["type"],
        old_observation=r["M_old"],
        new_observation=r["M_new"],
        explanation=r["explanation"],
    )
