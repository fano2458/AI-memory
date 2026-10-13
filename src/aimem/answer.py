import os
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from aimem.prompts import (
    PAPER_ANSWER_PROMPT, BENCHMARK_ANSWER_PROMPT,
    STALE_QUESTION_SYSTEM, STALE_QUESTION_USER, STALE_TASK_SYSTEM, STALE_TASK_USER,
)

KS = [10, 20, 50, 200]
PAPER_ONE_LIST_PROMPT = PAPER_ANSWER_PROMPT.replace("\n\n    Memories for user {speaker_2_user_id}:\n\n    {speaker_2_memories}", "")
gemma = None


def generate(messages, max_tokens):
    global gemma
    if gemma is None:
        gemma = OpenAI(base_url=os.environ["GEMMA_URL"], api_key="empty")
    start = time.time()
    response = gemma.chat.completions.create(model=os.environ["GEMMA_MODEL"], messages=messages, temperature=0, max_tokens=max_tokens)
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


def locomo_row(q, k, output, memory_ids, usage, prompt):
    return {
        "question_id": q.question_id,
        "category": q.category,
        "k": k,
        "question": q.text,
        "gold": q.answer,
        "adversarial_answer": q.extra.get("adversarial_answer"),
        "answer": output.rsplit("ANSWER:", 1)[-1].strip() if prompt == "benchmark" else output.strip(),
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


def answer_locomo(conversation, q, found, k, prompt):
    if prompt == "benchmark":
        messages = [{"role": "user", "content": benchmark_prompt(conversation, q, found, k)}]
    else:
        messages = [{"role": "system", "content": paper_prompt(conversation, q, found, k)}]
    output, usage = generate(messages, 4096)
    return locomo_row(q, k, output, [m[0] for memories in found.values() for m in memories[:k]], usage, prompt)


def answer_stale(conversation, q, found, k, prompt):
    memories = sorted((m for ms in found.values() for m in ms[:k]), key=lambda m: day(m[1]))
    history = "\n".join(f"({date}) {text}" for _, date, text in memories)
    system, user = (STALE_TASK_SYSTEM, STALE_TASK_USER) if q.category == "dim3" else (STALE_QUESTION_SYSTEM, STALE_QUESTION_USER)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user.format(history=history, query=q.text)}]
    output, usage = generate(messages, 2048)
    return stale_row(conversation, q, k, output.strip(), [m[0] for m in memories], usage)


def answer_all(conversation, retrieve, prompt):
    one = answer_locomo if conversation.benchmark == "locomo" else answer_stale
    jobs = []
    for q in conversation.questions:
        found = retrieve(q.text)
        jobs += [(q, found, k) for k in KS]
    with ThreadPoolExecutor(16) as pool:
        return list(pool.map(lambda job: one(conversation, *job, prompt), jobs))
