import re
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import orjson

DATA = Path(__file__).resolve().parents[2] / "data" / "external"
ABSTAIN = "Not mentioned in the conversation"


@dataclass
class Turn:
    turn_id: str
    speaker: str
    text: str


@dataclass
class Session:
    session_id: str
    index: int
    date: datetime
    turns: list[Turn]
    source_id: str | None = None


@dataclass
class Question:
    question_id: str
    text: str
    category: str
    answer: str | None
    abstention: bool
    date: datetime
    evidence_turns: list[str]
    evidence_sessions: list[str]
    extra: dict = field(default_factory=dict)


@dataclass
class Conversation:
    conversation_id: str
    benchmark: str
    speakers: list[str]
    sessions: list[Session]
    questions: list[Question]
    extra: dict = field(default_factory=dict)


def load_locomo():
    raw = orjson.loads((DATA / "locomo" / "locomo10.json").read_bytes())

    conversations = []
    for r in raw:
        c = r["conversation"]
        numbers = sorted(int(k.split("_")[1]) for k in c if re.fullmatch(r"session_\d+", k))

        sessions = []
        for i, n in enumerate(numbers):
            turns = []
            for turn in c[f"session_{n}"]:
                text = turn["text"]
                if turn.get("blip_caption"):
                    text += f" [shares a photo: {turn['blip_caption']}]"
                turns.append(Turn(turn["dia_id"], turn["speaker"], text))
            date = datetime.strptime(c[f"session_{n}_date_time"], "%I:%M %p on %d %B, %Y")
            sessions.append(Session(f"session_{n}", i, date, turns))

        session_of = {t.turn_id: s.session_id for s in sessions for t in s.turns}
        index_of = {s.session_id: s.index for s in sessions}

        questions = []
        for j, q in enumerate(r["qa"]):
            evidence = []
            for e in q["evidence"]:
                for piece in re.split(r"[;\s]+", e):
                    if piece in session_of and piece not in evidence:
                        evidence.append(piece)

            questions.append(Question(
                question_id=f"{r['sample_id']}:q{j}",
                text=q["question"],
                category=str(q["category"]),
                answer=str(q["answer"]) if "answer" in q else ABSTAIN,
                abstention="answer" not in q,
                date=sessions[-1].date,
                evidence_turns=evidence,
                evidence_sessions=sorted({session_of[t] for t in evidence}, key=index_of.get),
                extra={"adversarial_answer": q["adversarial_answer"]} if "adversarial_answer" in q else {},
            ))

        conversations.append(Conversation(
            conversation_id=r["sample_id"],
            benchmark="locomo",
            speakers=[c["speaker_a"], c["speaker_b"]],
            sessions=sessions,
            questions=questions,
        ))

    return conversations


def load_longmemeval(tier="s"):
    name = "longmemeval_oracle.json" if tier == "oracle" else f"longmemeval_{tier}_cleaned.json"
    raw = orjson.loads((DATA / "longmemeval" / name).read_bytes())
    fmt = "%Y/%m/%d (%a) %H:%M"

    conversations = []
    for r in raw:
        sessions, evidence_turns = [], []
        for i, (source_id, date, session) in enumerate(zip(r["haystack_session_ids"], r["haystack_dates"], r["haystack_sessions"])):
            turns = []
            for j, turn in enumerate(session):
                turn_id = f"s{i}:t{j}"
                turns.append(Turn(turn_id, turn["role"], turn["content"]))
                if str(turn.get("has_answer")) == "True":
                    evidence_turns.append(turn_id)
            sessions.append(Session(f"s{i}", i, datetime.strptime(date, fmt), turns, source_id))

        gold = set(r["answer_session_ids"])
        question = Question(
            question_id=r["question_id"],
            text=r["question"],
            category=r["question_type"],
            answer=str(r["answer"]),
            abstention=r["question_id"].endswith("_abs"),
            date=datetime.strptime(r["question_date"], fmt),
            evidence_turns=evidence_turns,
            evidence_sessions=[s.session_id for s in sessions if s.source_id in gold],
        )

        conversations.append(Conversation(
            conversation_id=r["question_id"],
            benchmark=f"longmemeval-{tier}",
            speakers=["user", "assistant"],
            sessions=sessions,
            questions=[question],
        ))

    return conversations


def load_stale():
    raw = orjson.loads((DATA / "stale" / "T1_T2_400_FULL.json").read_bytes())

    conversations = []
    for r in raw:
        sessions = []
        for i, (date, session) in enumerate(zip(r["timestamps"], r["haystack_session"])):
            turns = [Turn(f"s{i}:t{j}", turn["role"], turn["content"]) for j, turn in enumerate(session)]
            sessions.append(Session(f"s{i}", i, datetime.strptime(date, "%Y-%m-%d %H:%M"), turns))

        old, new = r["relevant_session_index"]
        questions = [
            Question(
                question_id=f"{r['uid']}:{dim}",
                text=r["probing_queries"][f"{dim}_query"],
                category=dim,
                answer=None,
                abstention=False,
                date=sessions[-1].date,
                evidence_turns=[],
                evidence_sessions=[f"s{old}", f"s{new}"],
            )
            for dim in ("dim1", "dim2", "dim3")
        ]

        conversations.append(Conversation(
            conversation_id=r["uid"],
            benchmark="stale",
            speakers=["user", "assistant"],
            sessions=sessions,
            questions=questions,
            extra={
                "type": r["type"],
                "old_observation": r["M_old"],
                "new_observation": r["M_new"],
                "explanation": r["explanation"],
            },
        ))

    return conversations


def stale_sample(per_type=50):
    conversations = load_stale()
    picked = []
    for kind in ["T1", "T2"]:
        group = [c for c in conversations if c.extra["type"] == kind]
        picked += random.Random(0).sample(group, per_type)
    return picked
