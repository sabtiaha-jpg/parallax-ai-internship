import pytest

from src.chunking import chunk_text


def test_empty_text():
    assert chunk_text("") == []


def test_short_text():
    text = "This is a short document."

    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_multiple_chunks():
    text = "A" * 1000

    chunks = chunk_text(
        text,
        chunk_size=500,
        overlap=100
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk) <= 500


def test_overlap():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    chunks = chunk_text(
        text,
        chunk_size=10,
        overlap=2
    )

    assert chunks[0][-2:] == chunks[1][:2]


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_text(
            "Some text",
            chunk_size=0,
            overlap=0
        )


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text(
            "Some text",
            chunk_size=100,
            overlap=100
        )