import os
import json
import asyncio
import logging
from datetime import timezone

import httpx

from aimem.llm import load_env

os.environ["GRAPHITI_TELEMETRY_ENABLED"] = "false"
os.environ.setdefault("SEMAPHORE_LIMIT", "4")
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
load_env()

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF, NODE_HYBRID_SEARCH_RRF
from graphiti_core.utils.maintenance import node_operations

NAME = "graphiti"
PROMPT = "zep"
FACT_SEARCH = EDGE_HYBRID_SEARCH_RRF.model_copy(update={"limit": 200})
ENTITY_SEARCH = NODE_HYBRID_SEARCH_RRF.model_copy(update={"limit": 200})
summarize = node_operations._process_summary_flight


async def summarize_or_keep(llm_client, nodes, *args, **kwargs):
    limit, llm_client.max_tokens = llm_client.max_tokens, 4096
    try:
        await summarize(llm_client, nodes, *args, **kwargs)
    except json.JSONDecodeError:
        print("summary rewrite failed, kept old summaries:", [node.name for node in nodes], flush=True)
    finally:
        llm_client.max_tokens = limit


node_operations._process_summary_flight = summarize_or_keep


class BGEEmbedder(EmbedderClient):
    async def create(self, input_data):
        text = input_data if isinstance(input_data, str) else " ".join(input_data)
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(os.environ["EMBED_URL"], json={"text": text or " "})
        response.raise_for_status()
        return response.json()["vector"]

    async def create_batch(self, input_data_list):
        return [await self.create(text) for text in input_data_list]


class Store:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.loop = asyncio.new_event_loop()
        config = LLMConfig(api_key="empty", model=os.environ["GEMMA_MODEL"], small_model=os.environ["GEMMA_MODEL"],
                           base_url=os.environ["GEMMA_URL"], temperature=0)
        self.graphiti = Graphiti(
            os.environ["NEO4J_URI"], os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"],
            llm_client=OpenAIGenericClient(config=config),
            embedder=BGEEmbedder(),
            cross_encoder=OpenAIRerankerClient(config=config),
        )
        self.run(self.graphiti.build_indices_and_constraints())

    def run(self, coroutine):
        return self.loop.run_until_complete(coroutine)


def open_store(run_dir):
    return Store(run_dir)


def close(store):
    store.run(store.graphiti.close())
    store.loop.close()


def day(when):
    return when.strftime("%d %B %Y") if when else "date unknown"


def query(store, cypher, group):
    return store.run(store.graphiti.driver.execute_query(cypher, params={"group": group}))[0]


def ingest(store, conversation, run_dir):
    group = conversation.conversation_id
    done = {r["name"] for r in query(store, "MATCH (e:Episodic {group_id: $group}) RETURN e.name AS name", group)}

    with open(run_dir / "turns.txt", "a", encoding="utf-8") as file:
        file.write(f"=== {group}\n")
        for session in conversation.sessions:
            date = session.date.strftime("%d %B %Y")
            file.write(f"--- {session.session_id} {date}\n")

            for turn in session.turns:
                if turn.turn_id in done:
                    continue
                file.write(f"{turn.turn_id} {turn.speaker}: {turn.text[:70]}\n")
                result = store.run(store.graphiti.add_episode(
                    name=turn.turn_id,
                    episode_body=f"{turn.speaker}: {turn.text}",
                    source_description=conversation.benchmark,
                    reference_time=session.date.replace(tzinfo=timezone.utc),
                    source=EpisodeType.message,
                    group_id=group,
                ))
                for edge in result.edges:
                    event = "INVALIDATE" if edge.invalid_at else "ADD"
                    file.write(f"     {edge.uuid[:8]} {event:10s} {edge.fact} ({day(edge.valid_at)} - {day(edge.invalid_at) if edge.invalid_at else 'present'})\n")

    names = {r["uuid"]: r["name"] for r in query(store, "MATCH (e:Episodic {group_id: $group}) RETURN e.uuid AS uuid, e.name AS name", group)}
    edges = query(store, "MATCH ()-[r:RELATES_TO {group_id: $group}]->() RETURN r.uuid AS uuid, r.episodes AS episodes, r.invalid_at AS invalid_at", group)
    provenance = {
        r["uuid"]: [([names[e] for e in r["episodes"] if e in names], "INVALIDATE" if r["invalid_at"] else "ADD")]
        for r in edges
    }
    with open(run_dir / "provenance.json", "w") as file:
        json.dump(provenance, file, indent=2)


def retrieve(store, conversation, query):
    group = [conversation.conversation_id]
    edges = store.run(store.graphiti.search_(query, config=FACT_SEARCH, group_ids=group)).edges
    nodes = store.run(store.graphiti.search_(query, config=ENTITY_SEARCH, group_ids=group)).nodes
    facts = [(e.uuid, e.fact, f"{day(e.valid_at)} - {day(e.invalid_at) if e.invalid_at else 'present'}") for e in edges]
    entities = [(n.uuid, n.name, n.summary) for n in nodes]
    return {"facts": facts, "entities": entities}
