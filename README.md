# AI-memory

Conflict-aware temporal memory for stateful LLM agents.

## Setup

Each memory system runs in its own environment:

```bash
uv venv envs/mem0-classic --python 3.11
uv pip install --python envs/mem0-classic/bin/python "mem0ai==1.0.11" orjson
uv venv envs/mem0-addonly --python 3.11
uv pip install --python envs/mem0-addonly/bin/python "mem0ai==2.2.1" orjson fastembed spacy pip
envs/mem0-addonly/bin/python -m spacy download en_core_web_sm
uv venv envs/cupmem --python 3.11
uv pip install --python envs/cupmem/bin/python -r external/stale/cup_mem/requirements.txt orjson
uv venv envs/graphiti --python 3.11
uv pip install --python envs/graphiti/bin/python "graphiti-core==0.30.2" orjson requests httpx
```

Graphiti uses Neo4j 5.26 in Docker (ports on localhost only; credentials in `.env` as `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`):

```bash
docker volume create neo4j-graphiti-data
docker run -d --name neo4j-graphiti --restart unless-stopped -p 127.0.0.1:7474:7474 -p 127.0.0.1:7687:7687 \
  -e NEO4J_AUTH="neo4j/$NEO4J_PASSWORD" -v neo4j-graphiti-data:/data neo4j:5.26-community
```

CUPMem comes from the STALE repository: clone https://github.com/icedreamc/STALE into `external/stale` and put `all-MiniLM-L6-v2` in `external/stale/cup_mem/models/`.

Endpoints, keys and model names live in `.env` (not committed): `OPENAI_API_KEY`, `GEMMA_URL`, `GEMMA_MODEL`, `QWEN_URL`, `QWEN_MODEL`, `EMBED_URL`, `JUDGE_MODEL`, `MEM0_TELEMETRY=False`.

## Data

Benchmarks are not committed. Download into `data/external/`:

```bash
curl -L -o data/external/stale/T1_T2_400_FULL.json \
  https://huggingface.co/datasets/STALEproj/STALE/resolve/main/T1_T2_400_FULL.json
curl -L -o data/external/longmemeval/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
curl -L -o data/external/locomo/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
```

## Running

Run from the repository root, in the system's environment:

```bash
envs/mem0-classic/bin/python scripts/run.py mem0 locomo
envs/mem0-addonly/bin/python scripts/run.py mem0 stale
envs/cupmem/bin/python scripts/run.py cupmem stale
envs/graphiti/bin/python scripts/run.py graphiti locomo
envs/mem0-classic/bin/python scripts/run.py fullcontext locomo --prompt benchmark
envs/mem0-classic/bin/python scripts/judge.py runs/stale/mem0-addonly
envs/mem0-classic/bin/python scripts/diagnose.py runs/locomo/mem0-classic
```

Results go to `runs/<benchmark>/<system>/<conversation or user>/`.

## Layout

| Path | Contents |
|---|---|
| `src/aimem/data.py` | records (`Conversation`, `Session`, `Turn`, `Question`), loaders for LoCoMo, LongMemEval and STALE, the fixed STALE sample |
| `src/aimem/prompts.py` | answer and judge prompts |
| `src/aimem/answer.py` | answer step for retrieval-based systems |
| `src/aimem/mem0.py`, `graphiti.py`, `cupmem.py`, `fullcontext.py` | memory systems |
| `src/aimem/llm.py` | `.env` loading and a cached LLM client |
| `scripts/run.py` | runs a system on a benchmark |
| `scripts/judge.py` | judges answers |
| `scripts/diagnose.py` | splits LoCoMo errors into storage, retrieval, memory and answering failures |

The first exploratory phase (retrieval experiments, scaling, synthetic generator) is preserved at git tag `exploration`.
