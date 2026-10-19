import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.vectorstore import get_vectorstore

RERANK_MODEL = "BAAI/bge-reranker-base"
MIN_RERANK_SCORE = 0.1  # 0-1 scale. Idhukku kammi irundha chunk irrelevant


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

    def vector_search(self, query: str, n: int = 20):
        return self.vs.similarity_search(query, k=n)

    def bm25_search(self, query: str, n: int = 20):
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)[:n]
        return [self.docs[i] for i in top]

    def rerank(self, query: str, docs, min_score: float = MIN_RERANK_SCORE):
        """Score kudutthu sort pannum, irrelevant chunks-ah neekidum."""
        if not docs:
            return docs
        if self.reranker is None:
            self.reranker = CrossEncoder(RERANK_MODEL)
        scores = self.reranker.predict([(query, d.page_content) for d in docs])
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        kept = [docs[i] for i in order if scores[i] >= min_score]
        return kept or [docs[order[0]]]  # onnume pass aagala-na best 1 mattum

    def search(self, query: str, k: int = 5, hybrid: bool = True, rerank: bool = True, candidates: int = 20):
        if not self.docs:  # Documents onnume illa-na empty list
            return []

        vec = self.vector_search(query, candidates)

        if hybrid:
            bm = self.bm25_search(query, candidates)
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