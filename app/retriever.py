import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.vectorstore import get_vectorstore

RERANK_MODEL = "BAAI/bge-reranker-base"


def tokenize(text: str):
    """Text-ah lowercase words-ah split pannum (BM25-ku thevai)."""
    return re.findall(r"\w+", text.lower())


def doc_key(doc: Document):
    """Ovvoru chunk-kum unique id (rendu search results-ah merge panna)."""
    m = doc.metadata
    return (m.get("source"), m.get("page"), m.get("start_index"))


def rrf(rank_lists, k: int = 60):
    """Reciprocal Rank Fusion: rendu ranked lists-ah oru list-ah merge pannum."""
    scores = {}
    for ranks in rank_lists:
        for position, key in enumerate(ranks):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + position + 1)
    return sorted(scores, key=scores.get, reverse=True)


class HybridRetriever:
    def __init__(self):
        self.vs = get_vectorstore()
        self.reranker = None  # first time thevai paduradhu varai load aagaadhu
        self.refresh()

    def refresh(self):
        """ChromaDB-la irukkura ella chunks-ah eduthu BM25 index-ah (re)build pannum."""
        data = self.vs.get()
        self.docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(data["documents"], data["metadatas"])
        ]
        self.bm25 = (
            BM25Okapi([tokenize(d.page_content) for d in self.docs])
            if self.docs
            else None
        )

    def list_sources(self):
        """Index aagirukkura file names list."""
        return sorted({d.metadata["source"] for d in self.docs})

    def vector_search(self, query: str, n: int = 20, sources: list[str] | None = None):
        flt = {"source": {"$in": sources}} if sources else None
        return self.vs.similarity_search(query, k=n, filter=flt)

    def bm25_search(self, query: str, n: int = 20, sources: list[str] | None = None):
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in order:
            d = self.docs[i]
            if sources and d.metadata["source"] not in sources:
                continue
            results.append(d)
            if len(results) == n:
                break
        return results

    def rerank(self, query: str, docs):
        """Question + chunk pair-ku score kudutthu, best chunks-ah mela kondu varum."""
        if not docs:
            return docs
        if self.reranker is None:
            self.reranker = CrossEncoder(RERANK_MODEL)
        scores = self.reranker.predict([(query, d.page_content) for d in docs])
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        return [docs[i] for i in order]

    def search(
        self,
        query: str,
        k: int = 8,
        hybrid: bool = True,
        rerank: bool = True,
        candidates: int = 20,
        sources: list[str] | None = None,
    ):
        if not self.docs:  # Documents onnume illa-na empty list
            return []

        vec = self.vector_search(query, candidates, sources)

        if hybrid:
            bm = self.bm25_search(query, candidates, sources)
            by_key = {doc_key(d): d for d in vec + bm}
            fused = rrf([[doc_key(d) for d in vec], [doc_key(d) for d in bm]])
            results = [by_key[key] for key in fused]
        else:
            results = vec

        if rerank:
            results = self.rerank(query, results[:candidates])

        return results[:k]


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "YARN"
    retriever = HybridRetriever()

    def show(title, docs):
        print(f"\n=== {title} ===")
        for d in docs:
            snippet = d.page_content[:90].replace("\n", " ")
            print(f" - {d.metadata['source']} (p{d.metadata.get('page', '-')}): {snippet}")

    print(f"Query: {query}")
    show("1. Vector only", retriever.search(query, hybrid=False, rerank=False))
    show("2. Hybrid", retriever.search(query, hybrid=True, rerank=False))
    show("3. Hybrid + Rerank", retriever.search(query, hybrid=True, rerank=True))