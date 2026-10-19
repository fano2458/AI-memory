import os
import sys
import json
import time
import threading
import shutil
import argparse
import importlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimem.llm import load_env

load_env()

from aimem.data import load_locomo, stale_sample
from aimem.answer import answer_all

SYSTEMS = {"mem0": "aimem.mem0", "graphiti": "aimem.graphiti", "cupmem": "aimem.cupmem", "fullcontext": "aimem.fullcontext"}
BENCHMARKS = {"locomo": load_locomo, "stale": stale_sample}
RUNS = Path(__file__).resolve().parents[1] / "runs"


def exit_with_parent(parent):
    def watch():
        while os.getppid() == parent:
            time.sleep(5)
        os._exit(1)
    threading.Thread(target=watch, daemon=True).start()


def run(module, conversation, run_dir, answers, prompt):
    system = importlib.import_module(module)
    if (run_dir / answers).exists():
        return "already done"

    if hasattr(system, "retrieve"):
        if (run_dir / "provenance.json").exists():
            memory = system.open_store(run_dir)
        else:
            shutil.rmtree(run_dir, ignore_errors=True)
            run_dir.mkdir(parents=True)
            memory = system.open_store(run_dir)
            system.ingest(memory, conversation, run_dir)
        rows = answer_all(conversation, lambda query: system.retrieve(memory, conversation, query), prompt)
        system.close(memory)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        rows = system.answer_all(conversation, run_dir, prompt)

    with open(run_dir / answers, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")
    return "done"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("system", choices=SYSTEMS)
    parser.add_argument("benchmark", choices=BENCHMARKS)
    parser.add_argument("--prompt", choices=["paper", "benchmark", "zep"], help="LoCoMo answer prompt; default: the system's own")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    system = importlib.import_module(SYSTEMS[args.system])
    prompt = args.prompt or system.PROMPT
    answers = "answers.jsonl" if prompt == system.PROMPT else f"answers-{prompt}.jsonl"
    root = RUNS / args.benchmark / system.NAME

    with ProcessPoolExecutor(args.workers, initializer=exit_with_parent, initargs=(os.getpid(),)) as pool:
        futures = {
            pool.submit(run, SYSTEMS[args.system], c, root / c.conversation_id, answers, prompt): c.conversation_id
            for c in BENCHMARKS[args.benchmark]()
        }
        for future in as_completed(futures):
            try:
                print(futures[future], future.result(), flush=True)
            except Exception as e:
                print(futures[future], "failed:", type(e).__name__, e, flush=True)
