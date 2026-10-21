import os
import re
import sys
import json
from pathlib import Path
from functools import partial

from openai import OpenAI

from aimem.answer import stale_row, locomo_row

external = Path(__file__).resolve().parents[2] / "external" / "stale"
sys.path.insert(0, str(external))

from cup_mem import CupMemEngine, LLMClient, PipelineThresholds, TraceConfig

NAME = "cupmem"
PROMPT = "own"


def item(conversation, owner, queries):
    locomo = conversation.benchmark == "locomo"
    return {
        "uid": conversation.conversation_id,
        "haystack_session": [
            [{"role": "user" if t.speaker == owner else "assistant", "content": f"{t.speaker}: {t.text}" if locomo else t.text} for t in s.turns]
            for s in conversation.sessions
        ],
        "timestamps": [s.date.strftime("%Y-%m-%d %H:%M") for s in conversation.sessions],
        "relevant_session_index": [] if locomo else [int(s[1:]) for s in conversation.questions[0].evidence_sessions],
        "probing_queries": queries,
    }


def run_engine(conversation, owner, queries, run_dir, suffix):
    llm = LLMClient(model=os.environ["GEMMA_MODEL"], api_key="empty", base_url=os.environ["GEMMA_URL"], cache_dir=run_dir / ".cache",
                    openai_cls=partial(OpenAI, timeout=180))
    engine = CupMemEngine(
        llm=llm,
        embedding_model_path=str(external / "cup_mem" / "models" / "all-MiniLM-L6-v2"),
        embedding_device="cpu",
        thresholds=PipelineThresholds(),
        trace_config=TraceConfig(),
    )
    result = engine.run_sample(item(conversation, owner, queries), sample_index=0, session_mode="full")
    (run_dir / f"trace{suffix}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / f"llm_calls{suffix}.json").write_text(json.dumps(llm.get_call_records(), ensure_ascii=False, indent=2), encoding="utf-8")
    logs = result["query_logs"]
    return {label: (log.get("answer", {}).get("answer", ""), round(1000 * log.get("elapsed_seconds", 0))) for label, log in logs.items()}


def answer_stale(conversation, run_dir):
    answers = run_engine(conversation, "user", {f"{q.category}_query": q.text for q in conversation.questions}, run_dir, "")
    rows = []
    for q in conversation.questions:
        answer, ms = answers[f"{q.category}_query"]
        rows.append(stale_row(conversation, q, "full", answer, [], {"in_tokens": None, "out_tokens": None, "ms": ms}))
    return rows


def answer_locomo(conversation, run_dir, prompt):
    first = {s: s.split()[0] for s in conversation.speakers}
    targets = {}
    for q in conversation.questions:
        named = [s for s in conversation.speakers if re.search(rf"\b{re.escape(first[s])}\b", q.text)]
        targets[q.question_id] = named if len(named) == 1 else conversation.speakers

    answers = {}
    for owner in conversation.speakers:
        queries = {q.question_id: q.text for q in conversation.questions if owner in targets[q.question_id]}
        answers[owner] = run_engine(conversation, owner, queries, run_dir, f"-{owner}")

    rows = []
    for q in conversation.questions:
        owners = targets[q.question_id]
        if len(owners) == 1:
            answer, ms = answers[owners[0]][q.question_id]
        else:
            answer = "\n".join(f"{o}'s memory: {answers[o][q.question_id][0]}" for o in owners)
            ms = sum(answers[o][q.question_id][1] for o in owners)
        rows.append(locomo_row(q, "full", answer, [], {"in_tokens": None, "out_tokens": None, "ms": ms}, prompt))
    return rows


def answer_all(conversation, run_dir, prompt):
    if conversation.benchmark == "stale":
        return answer_stale(conversation, run_dir)
    return answer_locomo(conversation, run_dir, prompt)
