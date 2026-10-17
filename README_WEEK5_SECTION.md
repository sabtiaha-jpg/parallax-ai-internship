## Week 5 — Evaluation & FastAPI

### Architecture
`POST /ask` → embed question → ChromaDB top-5 retrieval → relevance filtering → OpenRouter generation → semantic hallucination check → JSON response. Existing modules: `scripts/rag_pipeline.py` and `api/main.py`.

### Installation
From project root, activate the project's Python 3.11 virtual environment and run:

```powershell
python -m pip install "fastapi[standard]" pytest httpx
```

Also install existing dependencies (ChromaDB, sentence-transformers, openai, python-dotenv) following earlier project instructions. Keep `OPENROUTER_API_KEY` in `.env` (never commit it). The existing Wikipedia Chroma collection must be available at `data/chroma_db` under the name `wikipedia_chunks`.

### Start the service

```powershell
python -m uvicorn api.main:app --reload
```

Interactive documentation: http://127.0.0.1:8000/docs

Endpoints: `GET /`, `GET /health`, `POST /ask` with JSON `{"question":"What is artificial intelligence?"}`. Successful `/ask` returns `question`, `answer`, `sources`, and `latency_ms`. Request logs are JSON lines in `results/api_requests.log`, containing UTC timestamp, request ID, method, URL path, status and latency. Validation errors: 400 for whitespace-only questions, 422 for invalid request bodies, 500 for unexpected processing errors. Note: some OpenRouter failures are handled as text by the existing `generate_answer()`; they do not necessarily produce an HTTP 500.

### Run unit tests

```powershell
python -m pytest tests/test_api.py -v
```

The API tests mock vector retrieval and LLM calls. They verify 200/400/422/500, refusal behavior and request-ID headers without relying on a live model.

### Retrieval evaluation: Precision@5 and pooled Recall@5

```powershell
python -m evaluation.retrieval_eval prepare
```

Edit `evaluation/relevance_labels.json`: for **each** test question read the top-20 candidate previews, inspect full chunks if previews are ambiguous, and replace `"relevant_ids": null` with a list of IDs judged relevant. An empty list `[]` means none of the 20 candidates are relevant. Do not use guessed labels. Then run:

```powershell
python -m evaluation.retrieval_eval evaluate
```

The result is `results/retrieval_evaluation.json`. Precision@5 = relevant hits in the top 5 / 5. **Pooled** Recall@5 = relevant hits in the top 5 / manually judged relevant items in the top-20 pool. This is **not** full-corpus recall. Cases with no judged relevant items get null recall and are excluded from mean recall. Document the number of labeled cases and inspect disagreement/ambiguous queries.

### Generation evaluation and latency

```powershell
python -m evaluation.generation_eval
```

Results saved to `results/generation_evaluation.json`: per-question answer, refusals, semantic-support heuristic, keyword hit-rate heuristic, retrieval/generation/total latency and mean metrics. Edit `evaluation/questions.json` to add corpus-appropriate cases, and **manually review** the saved answers for correctness, context-groundedness, and refusals. Keyword matching and embedding similarity are only proxies and are not a validated ground-truth accuracy measure. The evaluation script uses existing RAG functions, including any upstream graceful error strings.

### Limitations and future improvements

- Automated API tests use mocks; validate live API separately in Swagger.
- Generation quality proxies can reward unsupported keyword matches or miss good paraphrases. Add blinded human judgments for factual correctness.
- Pooled Recall@5 is measured against labeled candidates, not every relevant chunk in the corpus.
- The legacy RAG function catches some provider errors and converts them to answer strings; proper 502/503 error mapping remains an improvement.
- Logs are JSON lines and include request metadata but omit sensitive question text.

### Week 5 deliverables

FastAPI service, structured request logging, error handling, API unit tests, labeled retrieval evaluation, generation evaluation, and saved reports. Report actual test output and metrics after running; never claim unexecuted tests or label preparation as completed evaluations.
