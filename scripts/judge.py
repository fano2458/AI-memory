import os
import re
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimem.llm import LLM, load_env
from aimem.data import ABSTAIN, load_stale
from aimem.prompts import (
    PAPER_JUDGE_PROMPT, STRICT_JUDGE_PROMPT, ADVERSARIAL_JUDGE_PROMPT, STALE_JUDGE_SYSTEM, STALE_JUDGE_USER,
    memconflict_prompts,
)

load_env()
llm = LLM(os.environ["JUDGE_MODEL"], max_tokens=4096)
DIMS = ["dim1", "dim2", "dim3"]
TYPES = ["dynamic", "static", "conditional"]
FLAG = {"dynamic": "update_awareness_and_order_consistency_score", "static": "conflict_recognition_score"}
RANK = {"dynamic": "updated_evidence_first_support_rank", "static": "truth_evidence_first_support_rank", "conditional": "correct_condition_evidence_first_support_rank"}


def label(prompt):
    text = llm.ask("", prompt)["text"]
    match = re.search(r'"label"\s*:\s*"(\w+)"', text)
    if match:
        return match.group(1).upper()
    words = re.findall(r"\b(CORRECT|WRONG|ABSTAIN|TRAP|OTHER|YES|NO)\b", text)
    return words[-1] if words else "UNPARSED"


def judge_locomo(row):
    if row["gold"] == ABSTAIN:
        row["adversarial"] = label(ADVERSARIAL_JUDGE_PROMPT.format(
            question=row["question"], adversarial_answer=row["adversarial_answer"], generated_answer=row["answer"]))
    else:
        args = {"question": row["question"], "gold_answer": row["gold"], "generated_answer": row["answer"]}
        row["paper"] = label(PAPER_JUDGE_PROMPT.format(**args))
        row["strict"] = label(STRICT_JUDGE_PROMPT.format(**args))
    return row


def judge_stale(job):
    conversation, k, answers = job
    queries = {q.category: q.text for q in conversation.questions}
    prompt = STALE_JUDGE_USER.format(
        old_info=conversation.extra["old_observation"],
        M_new=conversation.extra["new_observation"],
        explanation=conversation.extra["explanation"],
        dim1_q=queries["dim1"], dim1_a=answers["dim1"],
        dim2_q=queries["dim2"], dim2_a=answers["dim2"],
        dim3_q=queries["dim3"], dim3_a=answers["dim3"],
    )
    text = llm.ask(STALE_JUDGE_SYSTEM, prompt)["text"]
    try:
        result = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        passes = {d: bool(result[f"{d}_eval"]["pass"]) for d in DIMS}
    except (AttributeError, KeyError, TypeError, ValueError):
        passes = None
    return {"uid": conversation.conversation_id, "type": conversation.extra["type"], "k": k, "pass": passes, "judge": text}


def judge_memconflict(row):
    kind = row["category"]
    _, system, prompts = memconflict_prompts()
    text = llm.ask(system, prompts[kind].format(
        question=row["question"], gold_answer=row["gold"], model_answer=row["answer"],
        memories="\n".join(row["top_memories"]) or "No retrieved memories."))["text"]
    try:
        result = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        row["accuracy"] = float(result[f"{kind}_answer_accuracy"])
        row["flag"] = int(result[FLAG[kind]]) if kind in FLAG else None
        row["support_rank"] = int(result[RANK[kind]])
    except (AttributeError, KeyError, TypeError, ValueError):
        row["accuracy"] = None
    row["judge"] = text
    return row


def report_memconflict(rows):
    for k in sorted({r["k"] for r in rows}):
        group = [r for r in rows if r["k"] == k and r["accuracy"] is not None]
        line, scores = f"k={k:<4} questions={len(group):4d}", []
        for kind in TYPES:
            typed = [r for r in group if r["category"] == kind]
            if not typed:
                continue
            scores.append(100 * sum(r["accuracy"] for r in typed) / len(typed))
            line += f"  {kind} {scores[-1]:5.1f}"
            if kind in FLAG:
                line += f" (flag {100 * sum(r['flag'] for r in typed) / len(typed):5.1f})"
        overall = 100 * sum(r["accuracy"] for r in group) / len(group)
        hit = 100 * sum(r["support_rank"] > 0 for r in group) / len(group)
        print(f"{line}  mean of types {sum(scores) / len(scores):5.1f}  all {overall:5.1f}  top-3 hit {hit:5.1f}")
    unparsed = sum(r["accuracy"] is None for r in rows)
    if unparsed:
        print("unparsed:", unparsed)


def report_locomo(rows):
    by_k = defaultdict(list)
    for row in rows:
        by_k[row["k"]].append(row)

    for k, group in sorted(by_k.items(), key=lambda item: (isinstance(item[0], str), item[0])):
        answerable = [r for r in group if "paper" in r and r["category"] != "5"]
        adversarial = [r for r in group if "adversarial" in r]
        print(f"\nk={k}  questions={len(group)}")
        for name in ["paper", "strict"]:
            line = f"  {name:7s} cat1-4 {sum(r[name] == 'CORRECT' for r in answerable) / len(answerable):.3f}"
            for c in "1234":
                rs = [r for r in answerable if r["category"] == c]
                line += f"  c{c} {sum(r[name] == 'CORRECT' for r in rs) / len(rs):.3f}"
            print(line)
        labels = Counter(r["adversarial"] for r in adversarial)
        print("  cat5   ", "  ".join(f"{l} {n / len(adversarial):.3f}" for l, n in labels.most_common()))
        for name in ["paper", "strict"]:
            correct = sum(r.get(name) == "CORRECT" for r in group) + labels["ABSTAIN"]
            print(f"  {name:7s} all {len(group)}: {correct / len(group):.3f}")
        unparsed = sum("UNPARSED" in (r.get("paper"), r.get("strict"), r.get("adversarial")) for r in group)
        if unparsed:
            print("  unparsed:", unparsed)


def report_stale(rows):
    for k in sorted({r["k"] for r in rows}, key=lambda k: (isinstance(k, str), k)):
        group = [r for r in rows if r["k"] == k and r["pass"]]
        cells = []
        line = f"k={k:<4} users={len(group):3d} "
        for kind in ["T1", "T2"]:
            typed = [r for r in group if r["type"] == kind]
            for d, name in zip(DIMS, ["SR", "PR", "IPA"]):
                score = sum(r["pass"][d] for r in typed) / len(typed)
                cells.append(score)
                line += f" {kind} {name} {100 * score:5.1f}"
        print(f"{line}  overall {100 * sum(cells) / len(cells):5.1f}")
    unparsed = sum(r["pass"] is None for r in rows)
    if unparsed:
        print("unparsed:", unparsed)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    name = sys.argv[2] if len(sys.argv) > 2 else "answers"
    paths = sorted(root.glob(f"*/{name}.jsonl"))
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines()]

    if root.parent.name.startswith("stale"):
        conversations = {c.conversation_id: c for c in load_stale()}
        answers = defaultdict(dict)
        for row in rows:
            answers[row["uid"], row["k"]][row["dim"]] = row["answer"]
        jobs = [(conversations[uid], k, dims) for (uid, k), dims in answers.items()]
        judge, report = judge_stale, report_stale
    elif root.parent.name == "memconflict":
        jobs = rows
        judge, report = judge_memconflict, report_memconflict
    else:
        jobs = rows
        judge, report = judge_locomo, report_locomo
    print(len(jobs), "items from", len(paths), "conversations")

    with ThreadPoolExecutor(16) as pool:
        judged = list(pool.map(judge, jobs))

    with open(root / f"{name.replace('answers', 'judged')}.jsonl", "w", encoding="utf-8") as file:
        for row in judged:
            file.write(json.dumps(row) + "\n")

    report(judged)
