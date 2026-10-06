import numpy as np

import scripts.rag_pipeline as rag


# ============================================================
# RELEVANCE TESTS
# ============================================================

def test_relevant_documents():
    """Relevant documents should pass the distance threshold."""

    distances = [0.4160, 0.6990, 0.7659]

    assert rag.check_relevance(distances) is True


def test_irrelevant_documents():
    """Irrelevant documents should fail the distance threshold."""

    distances = [1.0917, 1.0953, 1.1224]

    assert rag.check_relevance(distances) is False


def test_empty_distances():
    """No retrieved documents should be considered irrelevant."""

    assert rag.check_relevance([]) is False


# ============================================================
# HALLUCINATION CHECK TESTS
# ============================================================

def test_refusal_is_safe():
    """A safe refusal should pass the hallucination check."""

    answer = (
        "I don't have enough information in the "
        "retrieved context to answer that question."
    )

    documents = [
        "Artificial intelligence is the study of intelligent machines."
    ]

    assert rag.hallucination_check(
        answer,
        documents,
    ) is True


def test_semantically_supported_answer(monkeypatch):
    """
    A generated answer that is semantically similar
    to the retrieved context should pass.
    """

    class FakeEmbeddingModel:

        def encode(
            self,
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ):
            # Same vector for answer and relevant context.
            if isinstance(text, list):
                return np.array([
                    [1.0, 0.0]
                    for _ in text
                ])

            return np.array([1.0, 0.0])

    monkeypatch.setattr(
        rag,
        "embedding_model",
        FakeEmbeddingModel(),
    )

    answer = (
        "Artificial intelligence is the study of "
        "intelligent machines."
    )

    documents = [
        "Artificial intelligence is the study of "
        "intelligent machines."
    ]

    assert rag.hallucination_check(
        answer,
        documents,
    ) is True


def test_unsupported_answer(monkeypatch):
    """
    An answer that is not semantically supported
    should fail the hallucination check.
    """

    class FakeEmbeddingModel:

        def encode(
            self,
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ):
            # Handle the list of retrieved documents.
            if isinstance(text, list):
                return np.array([
                    [0.0, 1.0]
                    for _ in text
                ])

            # The unsupported answer gets a different vector.
            if "capital of france" in text.lower():
                return np.array([1.0, 0.0])

            return np.array([0.0, 1.0])

    monkeypatch.setattr(
        rag,
        "embedding_model",
        FakeEmbeddingModel(),
    )

    answer = "The capital of France is Paris."

    documents = [
        "Artificial intelligence is the study of intelligent machines."
    ]

    assert rag.hallucination_check(
        answer,
        documents,
    ) is False