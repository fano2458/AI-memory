import re
import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimem.llm import load_env

load_env()

from aimem.answer import generate
from aimem.prompts import KEY_PROMPT, REPLACE_PROMPT

CANDIDATES = 20


def sessions(run_dir):
    full = {memory_id[:8]: memory_id for memory_id in json.loads((run_dir / "provenance.json").read_text())}
    found = []
    for line in (run_dir / "turns.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("--- "):
            found.append((line.split(" ", 2)[2], []))
        elif match := re.match(r"^     ([0-9a-f]{8}) ADD\s+(.*)$", line):
            found[-1][1].append((full[match[1]], match[2]))
    return [(date, memories) for date, memories in found if memories]


def ask(prompt):
    output, _ = generate([{"role": "user", "content": prompt}], 4096, json_mode=True)
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {}


def keys_for(memories, current):
    lines = "\n".join(f"[{i}] {text}" for i, (_, text) in enumerate(memories))
    given = ask(KEY_PROMPT.format(keys="\n".join(sorted(current)) or "(none yet)", memories=lines)).get("keys", {})
    if not isinstance(given, dict):
        given = {}
    keys = []
    for i in range(len(memories)):
        key = given.get(str(i))
        keys.append(" ".join(key.lower().split()) if isinstance(key, str) and key.strip() and key.strip().lower() != "null" else None)
    return keys


def number(value):
    text = str(value).strip().upper().lstrip("ON")
    return int(text) if text.isdigit() else None


def replaced(new, olds, texts):
    blocks = []
    for n, (memory_id, _) in enumerate(new):
        if olds[n]:
            older = "\n".join(f"  [O{o}] ({texts[old][0]}) {texts[old][1]}" for o, old in olds[n])
            blocks.append(f"[N{n}] ({texts[memory_id][0]}) {texts[memory_id][1]}\n{older}")
    if not blocks:
        return []
    pairs = ask(REPLACE_PROMPT.format(pairs="\n\n".join(blocks))).get("replaced", [])
    allowed = {(o, n) for n in range(len(new)) for o, _ in olds[n]}
    found = [(number(p.get("old")), number(p.get("new"))) for p in pairs if isinstance(p, dict)]
    return [pair for pair in found if pair in allowed]


def replay(run_dir):
    keys, status, texts, current = {}, {}, {}, {}
    for date, memories in sessions(run_dir):
        new = list(zip((memory_id for memory_id, _ in memories), keys_for(memories, current)))
        for memory_id, text in memories:
            texts[memory_id] = (date, text)
        numbered, olds = {}, []
        for memory_id, key in new:
            olds.append([])
            for old in current.get(key, [])[-CANDIDATES:] if key else []:
                numbered.setdefault(old, len(numbered))
                olds[-1].append((numbered[old], old))
        by_number = {o: old for old, o in numbered.items()}
        for o, n in replaced(new, olds, texts):
            old, (memory_id, key) = by_number[o], new[n]
            if old in status:
                continue
            status[old] = {"retired_since": date, "replaced_by": memory_id}
            current[keys[old]].remove(old)
        for memory_id, key in new:
            keys[memory_id] = key
            if key:
                current.setdefault(key, []).append(memory_id)
    (run_dir / "keys.json").write_text(json.dumps(keys, indent=2))
    (run_dir / "status.json").write_text(json.dumps(status, indent=2))
    return f"{len(keys)} memories, {sum(k is not None for k in keys.values())} with a key, {len(current)} keys, {len(status)} retired"


def run(run_dir):
    if (run_dir / "status.json").exists():
        return "already done"
    try:
        return replay(run_dir)
    except Exception as e:
        return f"failed: {type(e).__name__} {e}"


if __name__ == "__main__":
    root = Path(sys.argv[1])
    dirs = sorted(d for d in root.iterdir() if (d / "provenance.json").exists())
    with ThreadPoolExecutor(16) as pool:
        for run_dir, result in zip(dirs, pool.map(run, dirs)):
            print(run_dir.name, result, flush=True)
