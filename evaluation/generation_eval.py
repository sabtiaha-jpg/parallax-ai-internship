"""End-to-end RAG evaluation (automated *proxy* quality signals + latency).

This is not human-rated factual correctness. Keyword hits can miss valid
paraphrases or reward incorrect answers; inspect saved answers manually.
Run: python -m evaluation.generation_eval
"""
import json
import statistics
import time
from pathlib import Path
from typing import Any
from scripts import rag_pipeline as rag

INPUT = Path("evaluation/questions.json")
OUTPUT = Path("results/generation_evaluation.json")
REFUSAL = "I don't have enough information in the retrieved context to answer that question."


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one question through retrieval, generation, and safety checks.

    Args:
        case: Test case containing a ``question`` and optional
            ``expected_keywords`` list.

    Returns:
        A dictionary containing the answer, refusal and semantic-support
        indicators, keyword-hit-rate proxy, and latency measurements in ms.

    Note:
        Keyword matches and semantic support are heuristic signals, not
        verified factual correctness. For irrelevant retrieval, generation
        is skipped and the standard refusal is returned.
    """
    question = case["question"]
    start = time.perf_counter()
    retrieval_start = time.perf_counter()
    documents, distances = rag.retrieve_chunks(question)
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
    relevant = rag.check_relevance(distances)
    generation_ms = 0.0
    safe = None
    if not relevant:
        answer = REFUSAL
        safe = True
    else:
        generation_start = time.perf_counter()
        answer = rag.generate_answer(question, documents)
        generation_ms = (time.perf_counter() - generation_start) * 1000
        safe = rag.hallucination_check(answer, documents)
        if not safe:
            answer = REFUSAL
    keywords = case.get("expected_keywords", [])
    keyword_hit_rate = (sum(w.lower() in answer.lower() for w in keywords) / len(keywords)) if keywords else None
    refused = answer == REFUSAL or "don't have enough information" in answer.lower()
    return {"question": question, "answer": answer, "refused": refused,
            "keyword_hit_rate_proxy": keyword_hit_rate, "semantic_support_check": bool(safe),
            "retrieval_ms": round(retrieval_ms, 2), "generation_ms": round(generation_ms, 2),
            "total_ms": round((time.perf_counter() - start) * 1000, 2)}


def main() -> None:
    """Run all configured test cases and save a JSON evaluation report.

    Reads ``evaluation/questions.json`` and writes
    ``results/generation_evaluation.json``. Each case is evaluated
    independently, so a failed case is recorded without ending the run.
    The report includes per-case results and averages for successful cases.
    """
    cases = json.loads(INPUT.read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        try:
            result = evaluate_case(case)
            rows.append(result)
            print(f"{case['question']}: {result['total_ms']:.0f} ms")
        except Exception as exc:
            rows.append({"question": case["question"], "error": type(exc).__name__, "message": str(exc)})
            print(f"FAILED {case['question']}: {type(exc).__name__}")
    ok = [row for row in rows if "error" not in row]
    hits = [row["keyword_hit_rate_proxy"] for row in ok if row["keyword_hit_rate_proxy"] is not None]
    output = {
        "total_cases": len(cases), "successful_cases": len(ok),
        "mean_total_ms": round(statistics.mean(r["total_ms"] for r in ok), 2) if ok else None,
        "mean_keyword_hit_rate_proxy": round(statistics.mean(hits), 4) if hits else None,
        "methodology": "Keyword presence and embedding semantic-support heuristics are proxies, not validated factual accuracy. Review answers manually. Model errors may be returned as text by legacy generate_answer.",
        "results": rows,
    }
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
