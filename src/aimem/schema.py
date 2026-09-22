from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Turn:
    role: str
    content: str


@dataclass
class Session:
    index: int
    timestamp: str
    turns: list[Turn]

    @property
    def text(self) -> str:
        return "\n".join(f"{t.role}: {t.content}" for t in self.turns)


@dataclass
class Query:
    query_id: str
    dimension: str
    text: str


@dataclass
class Instance:
    instance_id: str
    sessions: list[Session]
    queries: list[Query]
    gold_sessions: list[int]
    conflict_type: str = ""
    old_observation: str = ""
    new_observation: str = ""
    explanation: str = ""

    @property
    def n_turns(self) -> int:
        return sum(len(s.turns) for s in self.sessions)


@dataclass
class Cost:
    reader_input_tokens: int = 0
    reader_output_tokens: int = 0
    index_tokens: int = 0
    n_embed_calls: int = 0
    n_reader_calls: int = 0
    n_judge_calls: int = 0
    retrieval_ms: float = 0.0
    reader_ms: float = 0.0
    judge_ms: float = 0.0


@dataclass
class Result:
    run_id: str
    benchmark: str
    instance_id: str
    query_id: str
    dimension: str
    baseline: str
    reader_model: str
    k: int
    conflict_type: str = ""
    retrieved_sessions: list[int] = field(default_factory=list)
    gold_sessions: list[int] = field(default_factory=list)
    prediction: str = ""
    verdict: str = ""
    score: float = 0.0
    cost: Cost = field(default_factory=Cost)
    git_commit: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
