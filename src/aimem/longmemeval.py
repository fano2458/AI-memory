import orjson
from pathlib import Path

from .schema import Instance, Query, Session, Turn

DIR = Path("data/external/longmemeval")
TIERS = {
    "oracle": DIR / "longmemeval_oracle.json",
    "s": DIR / "longmemeval_s_cleaned.json",
    "m": DIR / "longmemeval_m_cleaned.json",
}


def load(tier: str = "s", limit: int | None = None) -> list[Instance]:
    raw = orjson.loads(TIERS[tier].read_bytes())
    if limit:
        raw = raw[:limit]
    return [_parse(r) for r in raw]


def _parse(r: dict) -> Instance:
    sessions = [
        Session(
            index=i,
            timestamp=r["haystack_dates"][i],
            turns=[
                Turn(t["role"], t["content"], str(t.get("has_answer", "")) == "True") for t in turns
            ],
        )
        for i, turns in enumerate(r["haystack_sessions"])
    ]
    gold = set(r["answer_session_ids"])
    gold_sessions = [i for i, sid in enumerate(r["haystack_session_ids"]) if sid in gold]
    return Instance(
        instance_id=r["question_id"],
        sessions=sessions,
        queries=[Query(r["question_id"], r["question_type"], r["question"])],
        gold_sessions=gold_sessions,
        conflict_type=r["question_type"],
        answer=str(r["answer"]),
        query_date=r.get("question_date", ""),
    )
