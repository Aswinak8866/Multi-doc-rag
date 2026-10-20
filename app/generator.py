import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.retriever import HybridRetriever

load_dotenv()

LLM_MODEL = "openai/gpt-oss-120b"

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer ONLY using the context below. "
            "If the answer is not in the context, say: "
            "\"I don't know based on the provided documents.\" "
            "Be clear and concise. When asked to list items, list every item "
            "found in the context.\n\n"
            "Context:\n{context}",
        ),
        ("placeholder", "{history}"),
        ("human", "{question}"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's latest question as a standalone question, "
            "using the chat history to resolve words like 'it', 'that', 'they'. "
            "Do NOT answer it. If it is already standalone, return it unchanged. "
            "Return only the question.",
        ),
        ("placeholder", "{history}"),
        ("human", "{question}"),
    ]
)


class RAGPipeline:
    def __init__(self):
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY .env file-la illa. Check pannu.")
        self.retriever = HybridRetriever()
        self.llm = ChatGroq(model=LLM_MODEL, temperature=0)

    def _history_tuples(self, history):
        """Last 4 messages mattum eduthukkum."""
        return [(h["role"], h["content"]) for h in (history or [])[-4:]]

    def _standalone_question(self, question, history):
        """Follow-up question-ah thaniyaa purira maari maathum."""
        if not history:
            return question
        messages = REWRITE_PROMPT.format_messages(
            history=self._history_tuples(history), question=question
        )
        return self.llm.invoke(messages).content.strip()

    def ask(self, question, history=None, k=8, hybrid=True, rerank=True, sources=None):
        # 1. Follow-up-ah standalone question-ah maathu
        search_query = self._standalone_question(question, history)

        # 2. Relevant chunks search pannu (sources irundha andha files-la mattum)
        docs = self.retriever.search(
            search_query, k=k, hybrid=hybrid, rerank=rerank, sources=sources
        )
        if not docs:
            return {
                "answer": "No documents indexed yet. Upload a document first.",
                "sources": [],
                "search_query": search_query,
            }

        # 3. Context build pannu
        context = "\n\n".join(
            f"[{d.metadata['source']} | page {d.metadata.get('page', '-')}]\n{d.page_content}"
            for d in docs
        )

        # 4. LLM-ku anuppu (original question + history-oda)
        messages = ANSWER_PROMPT.format_messages(
            context=context,
            history=self._history_tuples(history),
            question=question,
        )
        answer = self.llm.invoke(messages).content

        # 5. Sources: same file + page repeat aagaama
        seen, source_list = set(), []
        for d in docs:
            page = d.metadata.get("page", "-")
            key = (d.metadata["source"], page)
            if key in seen:
                continue
            seen.add(key)
            source_list.append(
                {
                    "source": d.metadata["source"],
                    "page": page,
                    "snippet": d.page_content[:200].replace("\n", " "),
                }
            )

        return {"answer": answer, "sources": source_list, "search_query": search_query}


if __name__ == "__main__":
    rag = RAGPipeline()
    history = []
    print("Question type pannu ('exit' type panni nikkalaam)")

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = rag.ask(question, history=history)
        print(f"\n(Searched for: {result['search_query']})")
        print("\nAnswer:", result["answer"])
        print("\nSources:")
        for s in result["sources"]:
            print(f" - {s['source']} (page {s['page']})")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer"]})