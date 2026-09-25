import chromadb


def test_empty_collection():
    client = chromadb.Client()

    collection = client.create_collection(
        name="test_empty_collection"
    )

    assert collection.count() == 0


def test_add_document():
    client = chromadb.Client()

    collection = client.create_collection(
        name="test_add_document"
    )

    collection.add(
        ids=["1"],
        documents=["Artificial intelligence is a field of computer science."],
        embeddings=[[0.1, 0.2, 0.3]]
    )

    assert collection.count() == 1


def test_duplicate_id_handling():
    client = chromadb.Client()

    collection = client.create_collection(
        name="test_duplicate_id"
    )

    collection.add(
        ids=["1"],
        documents=["First document"],
        embeddings=[[0.1, 0.2, 0.3]]
    )

    # Using upsert prevents duplicate-ID failures
    collection.upsert(
        ids=["1"],
        documents=["Updated document"],
        embeddings=[[0.2, 0.3, 0.4]]
    )

    assert collection.count() == 1


def test_query_empty_collection():
    client = chromadb.Client()

    collection = client.create_collection(
        name="test_empty_query"
    )

    results = collection.get()

    assert len(results["ids"]) == 0