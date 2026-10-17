WEEK 5 COMPLETION PACKAGE

1. Extract into the ROOT of your existing parallax-ai-internship project (merge evaluation/ and tests/ folders). Do not replace your existing api/main.py or scripts/rag_pipeline.py.
2. Add the contents of README_WEEK5_SECTION.md to the END of your existing README.md.
3. Run: python -m pytest tests/test_api.py -v
4. Run: python -m evaluation.retrieval_eval prepare
5. Manually label evaluation/relevance_labels.json (replace null relevant_ids for all queries).
6. Run: python -m evaluation.retrieval_eval evaluate
7. Run: python -m evaluation.generation_eval
8. Review results/*.json and README, then git add/commit/push.

WARNING: Do not submit unverified scores or claim retrieval metrics complete until relevance labels have been reviewed.
