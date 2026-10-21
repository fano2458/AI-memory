import os
import re
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimem.llm import load_env

load_env()

from aimem.answer import generate
from aimem.prompts import REPLACE_PROMPT

CANDIDATES = 5


def writes(run_dir):
    full = {memory_id[:8]: memory_id for memory_id in json.loads((run_dir / "provenance.json").read_text())}
    found, date, fresh = [], None, True
    for line in (run_dir / "turns.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("--- "):
            date, fresh = line.split(" ", 2)[2], True
        elif match := re.match(r"^     ([0-9a-f]{8}) ADD\s+(.*)$", line):
            if fresh:
                found.append((date, []))
                fresh = False
            found[-1][1].append((full[match[1]], match[2]))
        elif not line.startswith(" "):
            fresh = True
    return found


def embed(text):
    for attempt in range(5):
        try:
            response = requests.post(os.environ["EMBED_URL"], json={"text": text or " "}, timeout=60)
            response.raise_for_status()
            return np.array(response.json()["vector"])
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(10 * (attempt + 1))


def ask(prompt):
    output, _ = generate([{"role": "user", "content": prompt}], 4096, json_mode=True)
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {}


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
    decisions = ask(REPLACE_PROMPT.format(pairs="\n\n".join(blocks))).get("decisions", [])
    allowed = {(o, n) for n in range(len(new)) for o, _ in olds[n]}
    found = [(number(d.get("old")), number(d.get("new")), d.get("label")) for d in decisions if isinstance(d, dict) and d.get("label") in ("replaced", "affected")]
    return [(o, n, label) for o, n, label in found if (o, n) in allowed]


def replay(run_dir):
    status, texts, vectors, current = {}, {}, {}, []
    for date, memories in writes(run_dir):
        for memory_id, text in memories:
            texts[memory_id] = (date, text)
            vectors[memory_id] = embed(text)
        matrix = np.array([vectors[old] for old in current]) if current else None
        numbered, olds = {}, []
        for memory_id, _ in memories:
            olds.append([])
            for i in np.argsort(-(matrix @ vectors[memory_id]))[:CANDIDATES] if current else []:
                old = current[i]
                numbered.setdefault(old, len(numbered))
                olds[-1].append((numbered[old], old))
        by_number = {o: old for old, o in numbered.items()}
        for o, n, label in replaced(memories, olds, texts):
            old = by_number[o]
            if old in status and (status[old]["label"] == "replaced" or label == "affected"):
                continue
            status[old] = {"retired_since": date, "replaced_by": memories[n][0], "label": label, "because": memories[n][1]}
            if label == "replaced":
                current.remove(old)
        current += [memory_id for memory_id, _ in memories]
    (run_dir / "status.json").write_text(json.dumps(status, indent=2))
    retired = sum(entry["label"] == "replaced" for entry in status.values())
    return f"{len(texts)} memories, {retired} retired, {len(status) - retired} doubtful"


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
