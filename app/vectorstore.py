from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.ingest import load_documents, load_file, split_documents

PERSIST_DIR = "chroma_db"
COLLECTION = "docs"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def get_embeddings():
    """Embedding model-ah oru thadava load panni reuse pannum."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore():
    """Already save panna ChromaDB-ah open pannum."""
    return Chroma(
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),
    )


def build_index():
    """data folder-la irukkura ella files-ah load, split, embed panni store pannum."""
    docs = load_documents()
    chunks = split_documents(docs)

    # pazhaya data irundha delete pannidum (duplicate varaadhu)
    get_vectorstore().delete_collection()

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION,
    )
    print(f"\nStored {len(chunks)} chunks in ChromaDB")
    return vs


def add_file(path) -> int:
    """Oru pudhu file-ah mattum ingest pannum (full rebuild vendaam)."""
    path = Path(path)
    docs = load_file(path)
    if not docs:
        return 0
    chunks = split_documents(docs)

    vs = get_vectorstore()
    vs.delete(where={"source": path.name})  # same file re-upload pannina pazhaya chunks neekidum
    vs.add_documents(chunks)
    return len(chunks)


if __name__ == "__main__":
    vs = build_index()

    query = "What is Apache Spark?"
    print(f"\nTest search: {query}\n")
    for i, doc in enumerate(vs.similarity_search(query, k=3), start=1):
        print(f"Result {i}: {doc.metadata['source']} (page {doc.metadata['page']})")
        print(doc.page_content[:150].replace("\n", " "))
        print()