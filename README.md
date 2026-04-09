# Adaptive RAG with Query Routing, Iterative Retrieval, and Self-Reflection

A runnable Python project for **Adaptive RAG** that lets the system decide whether to retrieve, perform iterative follow-up retrieval, and verify whether the answer is grounded in evidence.

## What this project implements

- **Query classifier / routing layer**
  - Routes simple questions to direct answering
  - Routes complex or time-sensitive questions to retrieval
- **Iterative retrieval**
  - First retrieval pass -> draft answer -> follow-up query -> second retrieval pass
- **Self-reflection / hallucination check**
  - Labels answers as `grounded`, `partially_grounded`, or `hallucinated`
  - Refuses to answer when support is too weak
- **Baseline comparison modes**
  - `vanilla`: retrieve once and answer
  - `graph`: retrieve + graph expansion
  - `adaptive`: route + iterative retrieval + reflection

## Design choice

This repo is designed to be **easy to run locally**:
- default retrieval uses **TF-IDF + cosine similarity**
- default reflection is a **lexical grounding heuristic**
- optional OpenAI API support is included for answer generation enhancement

So the project works **even without an API key**.

## Project structure

```text
adaptive_rag_project/
├── app/
│   ├── api/
│   ├── core/
│   ├── embeddings/
│   ├── evaluation/
│   ├── ingestion/
│   ├── llm/
│   ├── pipelines/
│   ├── retrieval/
│   ├── stores/
│   └── utils/
├── data/demo/
├── scripts/
├── tests/
├── requirements.txt
└── README.md
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

## Ingest demo data

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"data_dir":"./data/demo"}'
```

## Run a query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"How does adaptive RAG differ from vanilla RAG?","mode":"adaptive","top_k":4,"max_iterations":2}'
```

## API

- `GET /health`
- `POST /ingest`
- `POST /query`

### Query modes

- `vanilla`: single retrieval pass
- `graph`: retrieval + graph expansion
- `adaptive`: routing + iterative retrieval + reflection

## Example response fields

- `route`: `direct` or `rag`
- `iterations_used`
- `retrieved`
- `reflection`
- `follow_up_queries`

## Demo scripts

Run an interactive comparison:

```bash
python scripts/run_demo.py
```

Run a small benchmark-style comparison:

```bash
python scripts/evaluate_demo.py
```

## Benchmarks you can extend to

This repo ships with demo data and a tiny evaluation harness. You can extend it to:
- HotpotQA
- 2WikiMultihopQA
- FRAMES benchmark

Recommended upgrades:
- replace TF-IDF with dense embeddings
- replace heuristic router with an LLM router
- replace lexical reflection with an LLM verifier
- add BM25 + reranker
- add LangGraph / state-machine orchestration

## Suggested CV / GitHub framing

**Adaptive RAG system with query routing, iterative retrieval, and self-reflection for grounded QA.**

Or:

**Built an adaptive retrieval-augmented generation pipeline that dynamically routes queries, performs multi-step retrieval, and verifies grounding before answering.**
