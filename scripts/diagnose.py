import re
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judge import label
from aimem.data import load_locomo
from aimem.prompts import SUFFICIENCY_JUDGE_PROMPT

K = 200
STAGES = ["not stored", "deleted", "not retrieved", "label problem", "memory damaged", "answering failed", "unclear", "correct"]


def final_memories(run_dir, conversation):
    owner, date, final = None, None, {}
    for line in (run_dir / "turns.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("=== "):
            owner = line[4 + len(conversation.conversation_id) + 1:] or None
        elif line.startswith("--- "):
            date = line.split(" ", 2)[2]
        elif match := re.match(r"^     (\w{8}) (?:ADD|UPDATE)\s+(.*)$", line):
            final[match[1]] = (owner, date, match[2])
    return final


def stored_by_turn(run_dir, conversation, speaker):
    final = final_memories(run_dir, conversation)
    stored, deleted = defaultdict(set), defaultdict(set)
    for memory_id, events in json.loads((run_dir / "provenance.json").read_text()).items():
        target = deleted if any(event == "DELETE" for _, event in events) else stored
        owner = final[memory_id[:8]][0]
        for turns, _ in events:
            for turn in turns:
                if owner is None or speaker[turn] == owner:
                    target[turn].add(memory_id)
    return stored, deleted, final


def sufficient(question, gold, notes):
    return label(SUFFICIENCY_JUDGE_PROMPT.format(notes="\n".join(notes), question=question, gold_answer=gold))


def diagnose(case):
    row, memory_notes, turn_notes = case
    turns_ok = sufficient(row["question"], row["gold"], turn_notes)
    memories_ok = sufficient(row["question"], row["gold"], memory_notes)
    if "UNPARSED" in (turns_ok, memories_ok):
        stage = "unclear"
    elif turns_ok == "NO":
        stage = "label problem"
    elif memories_ok == "NO":
        stage = "memory damaged"
    else:
        stage = "answering failed"
    return {**row, "stage": stage, "evidence_memories": memory_notes, "evidence_turns_text": turn_notes}


if __name__ == "__main__":
    root = Path(sys.argv[1])
    name = sys.argv[2] if len(sys.argv) > 2 else "judged"
    rows = [json.loads(line) for line in (root / f"{name}.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["k"] == K and r["category"] != "5"]

    stages, cases = Counter(), []
    for conversation in load_locomo():
        turns = {t.turn_id: (s.date.strftime("%d %B %Y"), t) for s in conversation.sessions for t in s.turns}
        speaker = {turn_id: t.speaker for turn_id, (_, t) in turns.items()}
        evidence = {q.question_id: q.evidence_turns for q in conversation.questions}
        stored, deleted, final = stored_by_turn(root / conversation.conversation_id, conversation, speaker)

        for row in rows:
            need = evidence.get(row["question_id"])
            if not need:
                continue
            retrieved = set(row["memory_ids"])
            if not all(stored[t] for t in need):
                stages["deleted" if all(stored[t] or deleted[t] for t in need) else "not stored"] += 1
            elif not all(stored[t] & retrieved for t in need):
                stages["not retrieved"] += 1
            elif row["paper"] == "CORRECT":
                stages["correct"] += 1
            else:
                ids = sorted(set().union(*(stored[t] & retrieved for t in need)))
                memory_notes = [f"({final[m[:8]][1]}) {final[m[:8]][2]}" for m in ids]
                turn_notes = [f"({turns[t][0]}) {turns[t][1].speaker}: {turns[t][1].text}" for t in need]
                cases.append((row, memory_notes, turn_notes))

    with ThreadPoolExecutor(16) as pool:
        diagnosed = list(pool.map(diagnose, cases))
    stages.update(d["stage"] for d in diagnosed)

    with open(root / f"{name.replace('judged', 'diagnosed')}.jsonl", "w", encoding="utf-8") as file:
        for d in diagnosed:
            file.write(json.dumps(d) + "\n")

    total = sum(stages.values())
    print(f"k={K}, {total} category 1-4 questions with evidence")
    for stage in STAGES:
        print(f"  {stage:17s} {stages[stage]:5d}  {stages[stage] / total:.3f}")
