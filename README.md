# AI-memory

Conflict-aware temporal memory for stateful LLM agents.

## Setup

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
```

## Data

STALE (400 instances, 292 MB) is not committed. Download it into `data/external/stale/`:

```bash
curl -L -o data/external/stale/T1_T2_400_FULL.json \
  https://huggingface.co/datasets/STALEproj/STALE/resolve/main/T1_T2_400_FULL.json
```

## Run

```bash
python scripts/eval_retrieval.py --per-type 60
```

## Layout

| Path | Contents |
|---|---|
| `src/aimem/schema.py` | Shared records: `Instance`, `Session`, `Query`, `Result`, `Cost` |
| `src/aimem/stale.py` | STALE adapter — normalises the benchmark JSON |
| `src/aimem/retrieval.py` | Retrieval baselines: oracle, BM25 |
| `scripts/eval_retrieval.py` | Session-level CSR@k / ACH@k, no LLM needed |
