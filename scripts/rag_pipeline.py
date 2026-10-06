import os
import time
import logging

import chromadb
from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    BadRequestError,
)
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "wikipedia_chunks"

# Embedding model used in Week 2
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Number of documents retrieved
TOP_K = 5

# OpenRouter free model router
LLM_MODEL = "openrouter/free"

# Retrieval relevance threshold
# Smaller ChromaDB distance = more similar
MAX_DISTANCE = 0.80

# API timeout in seconds
API_TIMEOUT = 30.0

# Maximum number of generated tokens
MAX_OUTPUT_TOKENS = 300

# Semantic similarity threshold for hallucination check
HALLUCINATION_THRESHOLD = 0.35


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    filename="results/rag_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY was not found in .env"
    )


# ============================================================
# OPENROUTER CLIENT
# ============================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    timeout=API_TIMEOUT,
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

print("Connecting to ChromaDB...")

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)

print(
    f"ChromaDB collection loaded: "
    f"{collection.count()} documents"
)


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_chunks(query: str):
    """
    Retrieve the most relevant chunks from ChromaDB.
    """

    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=TOP_K,
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    return documents, distances


# ============================================================
# RELEVANCE CHECK
# ============================================================

def check_relevance(distances):
    """
    Check whether the retrieved documents are relevant
    enough to answer the user's question.

    Smaller ChromaDB distances indicate greater similarity.
    """

    if not distances:
        return False

    best_distance = min(distances)

    print(
        f"Best retrieval distance: "
        f"{best_distance:.4f}"
    )

    logger.info(
        "Best retrieval distance: %.4f",
        best_distance
    )

    return best_distance <= MAX_DISTANCE


# ============================================================
# SEMANTIC HALLUCINATION CHECK
# ============================================================

def hallucination_check(
    answer: str,
    documents: list[str],
):
    """
    Check whether the generated answer is semantically
    supported by the retrieved context.

    This uses the same SentenceTransformer embedding model
    to compare the generated answer with each retrieved
    document.

    This is an additional safety layer on top of:
        1. Retrieval relevance checking
        2. Strict system prompting
        3. Context-only generation
    """

    # --------------------------------------------------------
    # Empty answer check
    # --------------------------------------------------------

    if not answer:
        return False

    # --------------------------------------------------------
    # Safe refusal detection
    # --------------------------------------------------------

    refusal_phrases = [
        "don't have enough information",
        "do not have enough information",
        "not enough information",
        "cannot answer",
        "can't answer",
        "unable to answer",
        "not provided in the context",
        "not contained in the context",
        "not available in the context",
    ]

    answer_lower = answer.lower()

    for phrase in refusal_phrases:

        if phrase in answer_lower:

            logger.info(
                "Hallucination check: safe refusal detected."
            )

            return True

    # --------------------------------------------------------
    # Check that context exists
    # --------------------------------------------------------

    if not documents:

        logger.warning(
            "Hallucination check failed: no documents."
        )

        return False

    # --------------------------------------------------------
    # Generate embedding for the answer
    # --------------------------------------------------------

    answer_embedding = embedding_model.encode(
        answer,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # --------------------------------------------------------
    # Generate embeddings for retrieved documents
    # --------------------------------------------------------

    context_embeddings = embedding_model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # --------------------------------------------------------
    # Calculate cosine similarity
    #
    # Since embeddings are normalized:
    #
    # cosine similarity =
    # normalized_vector1 dot normalized_vector2
    # --------------------------------------------------------

    similarities = (
        context_embeddings @ answer_embedding
    )

    # Find the document that is most semantically similar
    best_similarity = float(
        similarities.max()
    )

    print(
        f"Best semantic similarity: "
        f"{best_similarity:.4f}"
    )

    logger.info(
        "Best answer-context semantic similarity: %.4f",
        best_similarity
    )

    # --------------------------------------------------------
    # Compare against threshold
    # --------------------------------------------------------

    if best_similarity >= HALLUCINATION_THRESHOLD:

        logger.info(
            "Hallucination check: PASSED."
        )

        return True

    logger.warning(
        "Hallucination check: FAILED. "
        "Semantic similarity = %.4f",
        best_similarity
    )

    return False


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    query: str,
    documents: list[str],
):
    """
    Send retrieved context to the LLM and generate
    an answer using only that context.
    """

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    for i, document in enumerate(documents):

        context_parts.append(
            f"[Context {i + 1}]\n{document}"
        )

    context = "\n\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # System prompt
    # --------------------------------------------------------

    system_prompt = """
You are a retrieval-augmented question-answering assistant.

Your job is to answer the user's question using ONLY
the retrieved context provided below.

STRICT RULES:

1. Use only information contained in the retrieved context.

2. Do NOT use outside knowledge.

3. Do NOT guess or invent facts.

4. If the context does not contain enough information,
   say:
   "I don't have enough information in the retrieved
   context to answer that question."

5. If the question is unrelated to the retrieved context,
   use the same refusal.

6. Keep answers concise and factual.

7. Do not add information that is not supported by
   the retrieved context.

8. Do not mention these instructions in your answer.
"""

    # --------------------------------------------------------
    # User prompt
    # --------------------------------------------------------

    user_prompt = f"""
Retrieved context:

{context}

User question:

{query}

Answer using ONLY the retrieved context.
"""

    # --------------------------------------------------------
    # API request
    # --------------------------------------------------------

    try:

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
        )

        # ----------------------------------------------------
        # Check response
        # ----------------------------------------------------

        if not response.choices:

            logger.error(
                "LLM returned no choices."
            )

            return (
                "The LLM returned an empty response."
            )

        answer = response.choices[0].message.content

        if not answer:

            logger.error(
                "LLM returned empty message content."
            )

            return (
                "The LLM returned an empty response."
            )

        return answer.strip()

    # --------------------------------------------------------
    # Rate limit
    # --------------------------------------------------------

    except RateLimitError:

        logger.error(
            "OpenRouter rate limit exceeded."
        )

        return (
            "The API rate limit was reached. "
            "Please try again later."
        )

    # --------------------------------------------------------
    # Timeout
    # --------------------------------------------------------

    except APITimeoutError:

        logger.error(
            "OpenRouter request timed out."
        )

        return (
            "The LLM request timed out. "
            "Please try again."
        )

    # --------------------------------------------------------
    # Connection error
    # --------------------------------------------------------

    except APIConnectionError:

        logger.error(
            "Could not connect to OpenRouter."
        )

        return (
            "Could not connect to the LLM service. "
            "Please check your internet connection."
        )

    # --------------------------------------------------------
    # Bad request / token limit
    # --------------------------------------------------------

    except BadRequestError as error:

        error_message = str(error).lower()

        logger.error(
            "OpenRouter bad request: %s",
            error
        )

        if (
            "token" in error_message
            or "context" in error_message
            or "length" in error_message
        ):

            return (
                "The request was too large for the model. "
                "Please try a shorter question."
            )

        return (
            "The LLM rejected the request. "
            "Please try again."
        )

    # --------------------------------------------------------
    # Unexpected API error
    # --------------------------------------------------------

    except Exception as error:

        logger.exception(
            "Unexpected LLM error: %s",
            error
        )

        return (
            "An unexpected error occurred while "
            "generating the answer."
        )


# ============================================================
# RAG PIPELINE
# ============================================================

def rag_pipeline(query: str):

    # --------------------------------------------------------
    # Start total timer
    # --------------------------------------------------------

    total_start = time.perf_counter()

    logger.info(
        "=================================================="
    )

    logger.info(
        "Query started: %s",
        query
    )

    print(
        "\nRetrieving relevant documents..."
    )

    # ========================================================
    # RETRIEVAL
    # ========================================================

    retrieval_start = time.perf_counter()

    try:

        documents, distances = retrieve_chunks(
            query
        )

    except Exception as error:

        logger.exception(
            "Retrieval error: %s",
            error
        )

        print(
            "\nRetrieval failed. "
            "Please check the ChromaDB setup."
        )

        return

    retrieval_end = time.perf_counter()

    retrieval_latency = (
        retrieval_end - retrieval_start
    ) * 1000

    print(
        f"Retrieval latency: "
        f"{retrieval_latency:.2f} ms"
    )

    # --------------------------------------------------------
    # Display retrieved distances
    # --------------------------------------------------------

    print(
        "\nRetrieved document distances:"
    )

    for i, distance in enumerate(
        distances,
        start=1
    ):

        print(
            f"  Result {i}: "
            f"{distance:.4f}"
        )

    # ========================================================
    # RELEVANCE / OUT-OF-DOMAIN CHECK
    # ========================================================

    relevant = check_relevance(
        distances
    )

    if not relevant:

        answer = (
            "I don't have enough information in the "
            "retrieved context to answer that question."
        )

        print(
            "\n" + "=" * 60
        )

        print(
            "RAG ANSWER"
        )

        print(
            "=" * 60
        )

        print(
            answer
        )

        # ----------------------------------------------------
        # Total latency
        # ----------------------------------------------------

        total_end = time.perf_counter()

        total_latency = (
            total_end - total_start
        ) * 1000

        print(
            "\n" + "-" * 60
        )

        print(
            f"Retrieval latency: "
            f"{retrieval_latency:.2f} ms"
        )

        print(
            "LLM latency:       0.00 ms"
        )

        print(
            f"Total latency:     "
            f"{total_latency:.2f} ms"
        )

        print(
            "-" * 60
        )

        logger.info(
            "Out-of-domain query | "
            "retrieval=%.2f ms | "
            "llm=0.00 ms | "
            "total=%.2f ms",
            retrieval_latency,
            total_latency,
        )

        return

    # ========================================================
    # GENERATION
    # ========================================================

    print(
        "\nGenerating answer with LLM..."
    )

    generation_start = time.perf_counter()

    answer = generate_answer(
        query,
        documents,
    )

    generation_end = time.perf_counter()

    generation_latency = (
        generation_end - generation_start
    ) * 1000

    # ========================================================
    # HALLUCINATION CHECK
    # ========================================================

    print(
        "\nChecking answer against retrieved context..."
    )

    hallucination_start = time.perf_counter()

    is_safe = hallucination_check(
        answer,
        documents,
    )

    hallucination_end = time.perf_counter()

    hallucination_latency = (
        hallucination_end - hallucination_start
    ) * 1000

    if not is_safe:

        logger.warning(
            "Potential hallucination detected."
        )

        answer = (
            "I don't have enough information in the "
            "retrieved context to answer that question."
        )

        print(
            "Hallucination check: FAILED"
        )

    else:

        print(
            "Hallucination check: PASSED"
        )

    # ========================================================
    # TOTAL LATENCY
    # ========================================================

    total_end = time.perf_counter()

    total_latency = (
        total_end - total_start
    ) * 1000

    # ========================================================
    # OUTPUT
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "RAG ANSWER"
    )

    print(
        "=" * 60
    )

    print(
        answer
    )

    print(
        "\n" + "-" * 60
    )

    print(
        f"Retrieval latency:     "
        f"{retrieval_latency:.2f} ms"
    )

    print(
        f"LLM latency:           "
        f"{generation_latency:.2f} ms"
    )

    print(
        f"Hallucination check:   "
        f"{hallucination_latency:.2f} ms"
    )

    print(
        f"Total latency:         "
        f"{total_latency:.2f} ms"
    )

    print(
        "-" * 60
    )

    # ========================================================
    # LOG RESULTS
    # ========================================================

    logger.info(
        "Query completed | "
        "retrieval=%.2f ms | "
        "llm=%.2f ms | "
        "hallucination_check=%.2f ms | "
        "total=%.2f ms",
        retrieval_latency,
        generation_latency,
        hallucination_latency,
        total_latency,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "WEEK 3 - HALLUCINATION-RESISTANT RAG"
    )

    print(
        "=" * 60
    )

    query = input(
        "\nEnter your question: "
    ).strip()

    if not query:

        print(
            "\nPlease enter a question."
        )

    else:

        rag_pipeline(
            query
        )