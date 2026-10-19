from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data")

LOADERS = {".pdf": PyPDFLoader, ".docx": Docx2txtLoader, ".txt": TextLoader}


def load_file(path: Path):
    """Oru file-ah load panni, metadata (source, page) add pannum."""
    loader_cls = LOADERS.get(path.suffix.lower())
    if loader_cls is None:
        print(f"Skipping unsupported file: {path.name}")
        return []

    kwargs = {"encoding": "utf-8"} if loader_cls is TextLoader else {}
    docs = loader_cls(str(path), **kwargs).load()

    for d in docs:
        d.metadata["source"] = path.name
        # PDF pages 0-la start aagum, adhai 1-la start aaga maathurom
        if "page" in d.metadata:
            d.metadata["page"] = int(d.metadata["page"]) + 1
    return docs


def load_documents(data_dir: Path = DATA_DIR):
    """data folder-la irukkura ella files-um load pannum."""
    docs = []
    for path in sorted(Path(data_dir).iterdir()):
        if path.is_file():
            loaded = load_file(path)
            print(f"Loaded {path.name}: {len(loaded)} page(s)")
            docs.extend(loaded)
    return docs


def split_documents(docs):
    """Periya text-ah chhota chunks-ah split pannum."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    print(f"\nTotal pages: {len(docs)}")
    print(f"Total chunks: {len(chunks)}")
    print("\n--- First chunk ---")
    print(chunks[0].page_content)
    print("\nMetadata:", chunks[0].metadata)