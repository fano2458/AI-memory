from concurrent.futures import ThreadPoolExecutor

from aimem.answer import generate, locomo_row
from aimem.prompts import FULL_CONTEXT_SYSTEM, FULL_CONTEXT_PROMPT, BENCHMARK_ANSWER_PROMPT

NAME = "full-context"
PROMPT = "paper"


def paper_context(conversation):
    lines = []
    for session in conversation.sessions:
        date = session.date.strftime("%I:%M %p on %d %B, %Y")
        for turn in session.turns:
            lines.append(f"{date} | {turn.speaker}: {turn.text}\n")
    return "".join(lines)


def benchmark_context(conversation):
    lines = ["The following memories are presented in chronological order (oldest to newest).", ""]
    for session in conversation.sessions:
        date = session.date.strftime("%A, %B %d, %Y")
        for turn in session.turns:
            lines.append(f"({date}) {turn.speaker}: {turn.text}")
    return "\n".join(lines)


def answer_all(conversation, run_dir, prompt):
    if conversation.benchmark != "locomo":
        raise ValueError("full context fits Gemma's window only on LoCoMo")
    context = benchmark_context(conversation) if prompt == "benchmark" else paper_context(conversation)
    reference = conversation.sessions[-1].date.strftime("%B %d, %Y")

    def one(q):
        if prompt == "benchmark":
            messages = [{"role": "user", "content": BENCHMARK_ANSWER_PROMPT.format(memories=context, question=q.text, reference_date=reference)}]
        else:
            messages = [
                {"role": "system", "content": FULL_CONTEXT_SYSTEM},
                {"role": "user", "content": FULL_CONTEXT_PROMPT.format(question=q.text, context=context)},
            ]
        output, usage = generate(messages, 4096)
        return locomo_row(q, "full", output, [], usage, prompt)

    with ThreadPoolExecutor(16) as pool:
        return list(pool.map(one, conversation.questions))
