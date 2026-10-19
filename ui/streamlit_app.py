import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Multi-Doc RAG", page_icon="📚")
st.title("📚 Multi-Document RAG Assistant")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Documents")
    files = st.file_uploader(
        "Upload PDF / DOCX / TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    if st.button("Ingest") and files:
        for f in files:
            with st.spinner(f"Ingesting {f.name}..."):
                try:
                    r = requests.post(
                        f"{API_URL}/upload",
                        files={"file": (f.name, f.getvalue())},
                        timeout=300,
                    )
                except requests.RequestException as e:
                    st.error(f"{f.name}: API connect aagala ({e})")
                    continue
            if r.ok:
                st.success(f"{f.name}: {r.json()['chunks']} chunks")
            else:
                st.error(f"{f.name}: {r.text}")

    try:
        names = requests.get(f"{API_URL}/sources", timeout=60).json()["sources"]
        st.caption("Indexed: " + ", ".join(names))
    except Exception:
        st.caption("Indexed documents list kidaikkala (API run aagudha check pannu)")

    st.divider()
    st.header("Settings")
    hybrid = st.checkbox("Hybrid search (BM25 + vector)", value=True)
    rerank = st.checkbox("Reranker", value=True)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- Chat state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources):
    with st.expander("Sources"):
        for s in sources:
            st.markdown(f"**{s['source']}** (page {s['page']})")
            st.caption(s["snippet"] + "...")


# Pazhaya messages-ah screen-la kaattu
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

# ---------------- New question ----------------
question = st.chat_input("Ask a question about your documents...")

if question:
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question,
                        "history": history,
                        "hybrid": hybrid,
                        "rerank": rerank,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                result = resp.json()
            except requests.RequestException as e:
                st.error(f"API error: {e}\n\nuvicorn server run aagudha nu check pannu.")
                st.stop()

        st.markdown(result["answer"])
        if result["sources"]:
            show_sources(result["sources"])

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )