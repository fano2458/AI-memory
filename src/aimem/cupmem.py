import os
import sys
import json
from pathlib import Path

from aimem.answer import stale_row

external = Path(__file__).resolve().parents[2] / "external" / "stale"
sys.path.insert(0, str(external))

from cup_mem import CupMemEngine, LLMClient, PipelineThresholds, TraceConfig

NAME = "cupmem"
PROMPT = "own"


def stale_item(conversation):
    return {
        "uid": conversation.conversation_id,
        "haystack_session": [[{"role": t.speaker, "content": t.text} for t in s.turns] for s in conversation.sessions],
        "timestamps": [s.date.strftime("%Y-%m-%d %H:%M") for s in conversation.sessions],
        "relevant_session_index": [int(s[1:]) for s in conversation.questions[0].evidence_sessions],
        "probing_queries": {f"{q.category}_query": q.text for q in conversation.questions},
    }


def answer_all(conversation, run_dir, prompt):
    if conversation.benchmark != "stale":
        raise ValueError("CUPMem is wired for STALE only")
    llm = LLMClient(model=os.environ["GEMMA_MODEL"], api_key="empty", base_url=os.environ["GEMMA_URL"], cache_dir=run_dir / ".cache")
    engine = CupMemEngine(
        llm=llm,
        embedding_model_path=str(external / "cup_mem" / "models" / "all-MiniLM-L6-v2"),
        embedding_device="cpu",
        thresholds=PipelineThresholds(),
        trace_config=TraceConfig(),
    )
    result = engine.run_sample(stale_item(conversation), sample_index=0, session_mode="full")
    (run_dir / "trace.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "llm_calls.json").write_text(json.dumps(llm.get_call_records(), ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for q in conversation.questions:
        log = result["query_logs"][f"{q.category}_query"]
        usage = {"in_tokens": None, "out_tokens": None, "ms": round(1000 * log.get("elapsed_seconds", 0))}
        rows.append(stale_row(conversation, q, "full", log.get("answer", {}).get("answer", ""), [], usage))
    return rows
