"""FastAPI endpoints and structured request logging for the Parallax RAG system."""

import time
import logging
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import Response

from scripts.rag_pipeline import (
    retrieve_chunks,
    check_relevance,
    generate_answer,
    hallucination_check,
)

# ============================================================
# LOGGING SETUP
# ============================================================

Path("results").mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

request_logger = logging.getLogger("api_requests")
request_logger.setLevel(logging.INFO)
request_logger.propagate = False

if not request_logger.handlers:
    handler = logging.FileHandler(
        "results/api_requests.log",
        encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    request_logger.addHandler(handler)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Parallax AI RAG API",
    description="RAG system with NLP metadata and evaluation",
    version="1.0.0"
)

REFUSAL = (
    "I don't have enough information in the "
    "retrieved context to answer that question."
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):
    """Validate incoming questions between 1 and 2,000 characters."""

    question: str = Field(min_length=1, max_length=2000)


# ============================================================
# STRUCTURED REQUEST LOGGING
# ============================================================

@app.middleware("http")
async def log_requests(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Log request metadata and elapsed time as JSON lines.

    Args:
        request: Incoming HTTP request.
        call_next: Passes the request to the next middleware or route.

    Returns:
        HTTP response with a unique X-Request-ID header.
    """
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code

        response.headers["X-Request-ID"] = request_id
        return response

    except Exception:
        logger.exception(
            "Unhandled request error | id=%s",
            request_id
        )
        raise

    finally:
        duration_ms = round(
            (time.perf_counter() - start) * 1000,
            2
        )

        log_entry = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": status_code,
            "duration_ms": duration_ms
        }

        request_logger.info(
            json.dumps(log_entry)
        )


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home() -> dict[str, str]:
    """Return the API welcome message and running status."""
    return {
        "message": "Welcome to Parallax AI RAG API",
        "status": "running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health() -> dict[str, str]:
    """Return the service health status."""
    return {
        "status": "healthy"
    }


# ============================================================
# ASK ENDPOINT
# ============================================================

@app.post("/ask")
def ask_question(
    request: QuestionRequest,
) -> dict[str, str | list[str] | float]:
    """Answer a question using retrieval-augmented generation.

    Args:
        request: Validated question from the JSON request body.

    Returns:
        The question, generated answer, source chunks, and latency in ms.

    Raises:
        HTTPException: 400 for whitespace-only questions or 500 for
            unexpected processing failures. Invalid request schema
            produces FastAPI's standard 422 response.
    """

    start = time.perf_counter()
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        # Step 1: Retrieve relevant documents
        documents, distances = retrieve_chunks(question)

        # Step 2: Check relevance
        if not check_relevance(distances):
            answer = REFUSAL
            documents = []

        else:
            # Step 3: Generate answer
            answer = generate_answer(
                question,
                documents
            )

            # Step 4: Hallucination check
            if not hallucination_check(
                answer,
                documents
            ):
                answer = REFUSAL

        # Step 5: Calculate latency
        latency_ms = round(
            (time.perf_counter() - start) * 1000,
            2
        )

        logger.info(
            "RAG request completed | latency_ms=%.2f",
            latency_ms
        )

        return {
            "question": question,
            "answer": answer,
            "sources": documents,
            "latency_ms": latency_ms
        }

    except Exception:
        logger.exception(
            "RAG API request failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Internal RAG processing error"
        )
