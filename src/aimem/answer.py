import os
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from aimem.prompts import (
    PAPER_ANSWER_PROMPT, BENCHMARK_ANSWER_PROMPT,
    STALE_QUESTION_SYSTEM, STALE_QUESTION_USER, STALE_TASK_SYSTEM, STALE_TASK_USER,
    ZEP_ANSWER_PROMPT, ZEP_CONTEXT_TEMPLATE, NEWEST_INSTRUCTION, DECIDE_PROMPT,
)

KS = [10, 20, 50, 200]
LADDER = ("newest", "decide", "status", "combined")
PAPER_ONE_LIST_PROMPT = PAPER_ANSWER_PROMPT.replace("\n\n    Memories for user {speaker_2_user_id}:\n\n    {speaker_2_memories}", "")
gemma = None


def generate(messages, max_tokens, json_mode=False):
    global gemma
    if gemma is None:
        gemma = OpenAI(base_url=os.environ["GEMMA_URL"], api_key="empty")
    start = time.time()
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = gemma.chat.completions.create(model=os.environ["GEMMA_MODEL"], messages=messages, temperature=0, max_tokens=max_tokens, **extra)
    output = response.choices[0].message.content or ""
    usage = {"in_tokens": response.usage.prompt_tokens, "out_tokens": response.usage.completion_tokens, "ms": round((time.time() - start) * 1000)}
    return output, usage


def day(date):
    return datetime.strptime(date, "%d %B %Y")


def paper_prompt(conversation, q, found, k):
    lists = {}
    for i, (owner, memories) in enumerate(found.items(), 1):
        lists[f"speaker_{i}_user_id"] = owner or " and ".join(conversation.speakers)
        lists[f"speaker_{i}_memories"] = json.dumps([f"{date}: {text}" for _, date, text in memories[:k]], indent=4)
    template = PAPER_ANSWER_PROMPT if len(found) > 1 else PAPER_ONE_LIST_PROMPT
    return template.format(question=q.text, **lists)


def benchmark_prompt(conversation, q, found, k):
    lines = ["The following memories are presented in chronological order (oldest to newest).", ""]
    for owner, memories in found.items():
        if owner:
            lines.append(f"Memories for {owner}:")
        dated = [(day(date), text) for _, date, text in memories[:k]]
        for when, text in sorted(dated, key=lambda m: m[0]):
            lines.append(f"({when:%A, %B %d, %Y}) {text}")
        lines.append("")
    reference = conversation.sessions[-1].date.strftime("%B %d, %Y")
    return BENCHMARK_ANSWER_PROMPT.format(memories="\n".join(lines).rstrip(), question=q.text, reference_date=reference)


def zep_context(found, k):
    facts = [f"  - {fact} ({span})" for _, fact, span in found["facts"][:k]]
    entities = [f"  - {name}: {summary}" for _, name, summary in found["entities"][:k]]
    return ZEP_CONTEXT_TEMPLATE.format(facts="\n".join(facts), entities="\n".join(entities))


def decide(q, memories):
    lines = "\n".join(f"[{i}] ({date}) {text}" for i, (_, date, text) in enumerate(memories))
    output, _ = generate([{"role": "user", "content": DECIDE_PROMPT.format(question=q.text, memories=lines)}], 4096, json_mode=True)
    try:
        pairs = json.loads(output)["outdated"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}
    valid = lambda p: (isinstance(p, dict) and all(isinstance(p.get(f), int) and 0 <= p[f] < len(memories) for f in ("id", "replaced_by"))
                       and p["id"] < p["replaced_by"])
    return {memories[p["id"]][0]: memories[p["replaced_by"]][1] for p in pairs if valid(p)}


def mark(memories, outdated):
    return [(i, date, f"[OUTDATED since {outdated[i]}] {text}" if i in outdated else text) for i, date, text in memories]


def outdated_for(q, memories, prompt, retired):
    stored = {m[0]: retired[m[0]]["retired_since"] for m in memories if m[0] in retired}
    if prompt == "decide":
        return decide(q, memories)
    if prompt == "status":
        return stored
    if prompt == "combined":
        return {**decide(q, memories), **stored}
    return {}


def locomo_row(q, k, output, memory_ids, usage, prompt):
    return {
        "question_id": q.question_id,
        "category": q.category,
        "k": k,
        "question": q.text,
        "gold": q.answer,
        "adversarial_answer": q.extra.get("adversarial_answer"),
        "answer": output.rsplit("ANSWER:", 1)[-1].strip() if prompt in ("benchmark", *LADDER) else output.strip(),
        "output": output,
        "memory_ids": memory_ids,
        **usage,
    }


def stale_row(conversation, q, k, answer, memory_ids, usage):
    return {
        "question_id": q.question_id,
        "uid": conversation.conversation_id,
        "type": conversation.extra["type"],
        "dim": q.category,
        "k": k,
        "question": q.text,
        "answer": answer,
        "memory_ids": memory_ids,
        **usage,
    }


def answer_locomo(conversation, q, found, k, prompt, retired):
    if prompt == "zep":
        messages = [{"role": "system", "content": ZEP_ANSWER_PROMPT.format(memories=zep_context(found, k), question=q.text)}]
    elif prompt == "benchmark":
        messages = [{"role": "user", "content": benchmark_prompt(conversation, q, found, k)}]
    elif prompt in LADDER:
        outdated = outdated_for(q, sorted((m for ms in found.values() for m in ms[:k]), key=lambda m: day(m[1])), prompt, retired)
        found = {owner: mark(ms[:k], outdated) for owner, ms in found.items()}
        messages = [{"role": "user", "content": f"{NEWEST_INSTRUCTION}\n\n{benchmark_prompt(conversation, q, found, k)}"}]
    else:
        messages = [{"role": "system", "content": paper_prompt(conversation, q, found, k)}]
    output, usage = generate(messages, 4096)
    row = locomo_row(q, k, output, [m[0] for memories in found.values() for m in memories[:k]], usage, prompt)
    if prompt in ("decide", "status", "combined"):
        row["outdated"] = list(outdated)
    return row


def answer_stale(conversation, q, found, k, prompt, retired):
    if "facts" in found:
        memories = found["facts"][:k] + found["entities"][:k]
        history = zep_context(found, k)
    else:
        memories = sorted((m for ms in found.values() for m in ms[:k]), key=lambda m: day(m[1]))
        outdated = outdated_for(q, memories, prompt, retired)
        history = "\n".join(f"({date}) {text}" for _, date, text in mark(memories, outdated))
    system, user = (STALE_TASK_SYSTEM, STALE_TASK_USER) if q.category == "dim3" else (STALE_QUESTION_SYSTEM, STALE_QUESTION_USER)
    if prompt in LADDER:
        system = f"{system} {NEWEST_INSTRUCTION}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user.format(history=history, query=q.text)}]
    output, usage = generate(messages, 2048)
    row = stale_row(conversation, q, k, output.strip(), [m[0] for m in memories], usage)
    if prompt in ("decide", "status", "combined"):
        row["outdated"] = list(outdated)
    return row


def answer_all(conversation, retrieve, prompt, ks=KS, retired=None):
    one = answer_locomo if conversation.benchmark == "locomo" else answer_stale
    jobs = []
    for q in conversation.questions:
        found = retrieve(q.text)
        jobs += [(q, found, k) for k in ks]
    with ThreadPoolExecutor(16) as pool:
        return list(pool.map(lambda job: one(conversation, *job, prompt, retired or {}), jobs))
