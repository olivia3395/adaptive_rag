<div align="center">

# 🧠 Adaptive RAG

**Query Routing · Iterative Retrieval · Self-Reflection**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-F7DF1E?style=flat-square)](LICENSE)
[![No API Key Required](https://img.shields.io/badge/API%20Key-Optional-brightgreen?style=flat-square)](##quickstart)

*A production-ready Python pipeline that decides when to retrieve, how many times to retrieve, and whether the answer is actually grounded in evidence.*

</div>


## ✨ What Makes This Different

Most RAG systems retrieve once and hope for the best. This one **thinks before, during, and after**.

| Feature | Vanilla RAG | This Project |
|---|:---:|:---:|
| Single retrieval pass | ✅ | ✅ |
| Query routing (skip retrieval when unnecessary) | ❌ | ✅ |
| Iterative follow-up retrieval | ❌ | ✅ |
| Hallucination / grounding check | ❌ | ✅ |
| Works without an API key | ❌ | ✅ |



## 🏗️ Architecture

```
                      ┌─────────────────────┐
         question ──► │   Query Classifier   │
                      └────────┬────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                                  ▼
        "simple fact"                     "complex / recent"
              │                                  │
              ▼                                  ▼
       Direct Answer                    ┌──────────────────┐
                                        │  Retrieval Pass 1 │
                                        └────────┬─────────┘
                                                 │
                                        ┌────────▼─────────┐
                                        │   Draft Answer    │
                                        └────────┬─────────┘
                                                 │
                                        ┌────────▼─────────┐
                                        │  Follow-up Query  │
                                        └────────┬─────────┘
                                                 │
                                        ┌────────▼─────────┐
                                        │  Retrieval Pass 2 │
                                        └────────┬─────────┘
                                                 │
                                        ┌────────▼─────────┐
                                        │  Self-Reflection  │◄── grounded?
                                        └────────┬─────────┘     partially?
                                                 │                hallucinated?
                                        ┌────────▼─────────┐
                                        │   Final Answer    │
                                        └──────────────────┘
```



## 📦 Project Structure

```
adaptive_rag/
├── app/
│   ├── api/              # FastAPI route handlers
│   ├── core/             # Config & shared types
│   ├── embeddings/       # TF-IDF / dense embedding adapters
│   ├── evaluation/       # Grounding scorer & metrics
│   ├── ingestion/        # Document loading & chunking
│   ├── llm/              # LLM abstraction (local + OpenAI)
│   ├── pipelines/        # vanilla · graph · adaptive
│   ├── retrieval/        # Cosine sim retriever
│   ├── stores/           # Vector & document stores
│   └── utils/            # Logging, text helpers
├── data/demo/            # Sample documents to get started
├── scripts/
│   ├── run_demo.py       # Interactive comparison
│   └── evaluate_demo.py  # Mini benchmark harness
├── tests/
├── requirements.txt
└── README.md
```



## 🚀 Quickstart

### 1 · Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

> **No API key needed.** The default pipeline uses TF-IDF retrieval and a lexical grounding heuristic — everything runs fully offline.

### 2 · Start the server

```bash
uvicorn app.main:app --reload
```

### 3 · Ingest demo documents

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"data_dir": "./data/demo"}'
```

### 4 · Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question":       "How does adaptive RAG differ from vanilla RAG?",
    "mode":           "adaptive",
    "top_k":          4,
    "max_iterations": 2
  }'
```



## 🌐 API Reference

### `GET /health`
Returns service status.

### `POST /ingest`
| Field | Type | Description |
|---|---|---|
| `data_dir` | `string` | Path to a directory of `.txt` / `.md` documents |

### `POST /query`
| Field | Type | Default | Description |
|---|---|---|---|
| `question` | `string` | — | The user's question |
| `mode` | `string` | `adaptive` | `vanilla` · `graph` · `adaptive` |
| `top_k` | `int` | `4` | Documents retrieved per pass |
| `max_iterations` | `int` | `2` | Maximum retrieval rounds |

### Response fields

```jsonc
{
  "answer":           "...",
  "route":            "direct | rag",
  "iterations_used":  2,
  "retrieved":        [...],
  "reflection":       "grounded | partially_grounded | hallucinated",
  "follow_up_queries": [...]
}
```



## 🔬 Query Modes

### `vanilla`
Single retrieval pass → answer. Fast baseline.

### `graph`
Retrieval + neighbour expansion via a document similarity graph. Better recall on related concepts.

### `adaptive` ⭐
Full pipeline: route → iterative retrieval → draft → follow-up → reflect → final answer. Best quality.



## 🧪 Demo Scripts

Run an **interactive side-by-side comparison** of all three modes:
```bash
python scripts/run_demo.py
```

Run the **mini benchmark harness** to compare precision / grounding scores:
```bash
python scripts/evaluate_demo.py
```



## 📊 Extending to Real Benchmarks

This repo ships with demo data and a lightweight eval harness. Swap in any QA dataset:

- [HotpotQA](https://hotpotqa.github.io/) — multi-hop reasoning
- [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop) — cross-document inference
- [FRAMES](https://github.com/google-deepmind/frames) — factual & reasoning evaluation



## 🔧 Recommended Upgrades

```
TF-IDF retrieval      →  Dense embeddings (e.g. BGE, E5, OpenAI ada)
Lexical router        →  LLM-based query classifier
Lexical reflection    →  LLM verifier (NLI or prompted)
Single retriever      →  BM25 + cross-encoder reranker
Ad-hoc orchestration  →  LangGraph state-machine
```



## 💼 GitHub / CV Framing

> *Built an adaptive retrieval-augmented generation pipeline that dynamically routes queries, performs multi-step retrieval, and verifies answer grounding before responding.*

Short version:
> *Adaptive RAG system with query routing, iterative retrieval, and self-reflection for grounded QA.*

<div align="center">



</div>
