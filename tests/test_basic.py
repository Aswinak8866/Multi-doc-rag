from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.api import app
from app.ingest import split_documents
from app.retriever import rrf, tokenize

client = TestClient(app)


def test_tokenize_lowercases_and_splits():
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_rrf_prefers_chunk_present_in_both_lists():
    fused = rrf([["a", "b"], ["b", "c"]])
    assert fused[0] == "b"


def test_split_documents_creates_multiple_chunks():
    doc = Document(page_content="word " * 1000, metadata={"source": "x.txt", "page": 1})
    chunks = split_documents([doc])
    assert len(chunks) > 1


def test_split_documents_keeps_metadata():
    doc = Document(page_content="word " * 1000, metadata={"source": "x.txt", "page": 1})
    chunks = split_documents([doc])
    assert all(c.metadata["source"] == "x.txt" for c in chunks)
    assert all("start_index" in c.metadata for c in chunks)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_rejects_unsupported_file_type():
    response = client.post("/upload", files={"file": ("virus.exe", b"abc")})
    assert response.status_code == 400