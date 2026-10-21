import os
import json
import time
import importlib.metadata
from pathlib import Path

import requests

from aimem.llm import load_env

VERSION = importlib.metadata.version("mem0ai")
ADDONLY = VERSION.startswith("2.")
NAME = "mem0-addonly" if ADDONLY else "mem0-classic"
PROMPT = "benchmark" if ADDONLY else "paper"
MAX_K = 200

state = Path(__file__).resolve().parents[2] / "runs" / f".mem0-{VERSION}"
state.mkdir(parents=True, exist_ok=True)
os.environ["MEM0_DIR"] = str(state)
load_env()

from mem0 import Memory
from mem0.embeddings.base import EmbeddingBase


class BGEEmbedder(EmbeddingBase):
    def embed(self, text, memory_action=None):
        for attempt in range(5):
            try:
                response = requests.post(os.environ["EMBED_URL"], json={"text": text or " "}, timeout=60)
                response.raise_for_status()
                return response.json()["vector"]
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(10 * (attempt + 1))


def facts_as_objects(generate):
    def wrapped(*args, **kwargs):
        response = generate(*args, **kwargs)
        try:
            data = json.loads(response)
        except (TypeError, ValueError):
            return response
        facts = data.get("memory") if isinstance(data, dict) else None
        if isinstance(facts, list) and any(isinstance(fact, str) for fact in facts):
            data["memory"] = [{"text": fact} if isinstance(fact, str) else fact for fact in facts]
            return json.dumps(data)
        return response
    return wrapped


def open_store(run_dir):
    config = {
        "llm": {"provider": "openai", "config": {
            "model": os.environ["GEMMA_MODEL"],
            "openai_base_url": os.environ["GEMMA_URL"],
            "api_key": "empty",
            "temperature": 0,
        }},
        "embedder": {"provider": "openai", "config": {
            "api_key": "empty",
            "embedding_dims": 1024,
        }},
        "vector_store": {"provider": "qdrant", "config": {
            "embedding_model_dims": 1024,
            "path": str(run_dir / "qdrant"),
            "collection_name": "memories",
        }},
        "history_db_path": str(run_dir / "history.db"),
    }
    memory = Memory.from_config(config)
    memory.embedding_model = BGEEmbedder()
    if ADDONLY:
        memory.llm.generate_response = facts_as_objects(memory.llm.generate_response)
    return memory


def close(memory):
    memory.vector_store.client.close()


def stores(conversation):
    cid = conversation.conversation_id
    if conversation.benchmark != "locomo":
        return {cid: "user"}
    if ADDONLY:
        return {cid: conversation.speakers[0]}
    return {f"{cid}-{s}": s for s in conversation.speakers}


def ingest(memory, conversation, run_dir):
    locomo = conversation.benchmark == "locomo"
    batch = 1 if locomo and ADDONLY else 2
    prefix = ADDONLY or not locomo

    provenance = {}
    with open(run_dir / "turns.txt", "w", encoding="utf-8") as file:
        for user_id, owner in stores(conversation).items():
            file.write(f"=== {user_id}\n")

            for session in conversation.sessions:
                date = session.date.strftime("%d %B %Y")
                file.write(f"--- {session.session_id} {date}\n")

                for i in range(0, len(session.turns), batch):
                    turns = session.turns[i:i + batch]
                    messages = []
                    for turn in turns:
                        content = f"{turn.speaker}: {turn.text}" if locomo else turn.text
                        if prefix:
                            content = f"[{date}] {content}"
                        messages.append({"role": "user" if turn.speaker == owner else "assistant", "content": content})
                        file.write(f"{turn.turn_id} {turn.speaker}: {turn.text[:70]}\n")

                    out = memory.add(messages, user_id=user_id, metadata={"timestamp": date})
                    for r in out["results"]:
                        file.write(f"     {r['id'][:8]} {r['event']:6s} {r['memory']}\n")
                        if r["event"] == "UPDATE":
                            file.write(f"              was: {r['previous_memory']}\n")
                        if r["event"] == "NONE":
                            continue
                        if r["id"] not in provenance:
                            provenance[r["id"]] = []
                        provenance[r["id"]].append(([t.turn_id for t in turns], r["event"]))

    with open(run_dir / "provenance.json", "w") as file:
        json.dump(provenance, file, indent=2)


def retrieve(memory, conversation, query):
    found = {}
    owners = stores(conversation)
    for user_id, owner in owners.items():
        if ADDONLY:
            out = memory.search(query, filters={"user_id": user_id}, top_k=MAX_K)
        else:
            out = memory.search(query, user_id=user_id, limit=MAX_K)
        found[owner if len(owners) > 1 else None] = [(r["id"], r["metadata"]["timestamp"], r["memory"]) for r in out["results"]]
    return found
