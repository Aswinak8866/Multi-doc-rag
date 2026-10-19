import logging
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.generator import RAGPipeline
from app.ingest import DATA_DIR
from app.vectorstore import add_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("api")

app = FastAPI(title="Multi-Document RAG API")

ALLOWED = {".pdf", ".docx", ".txt"}
_pipeline = None


def get_pipeline() -> RAGPipeline:
    """Pipeline-ah first request-la mattum load pannum (models load aaga neram aagum)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


class Message(BaseModel):
    role: str  # "user" illa "assistant"
    content: str


class AskRequest(BaseModel):
    question: str
    history: list[Message] = []
    hybrid: bool = True
    rerank: bool = True


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    name = Path(file.filename).name
    if Path(name).suffix.lower() not in ALLOWED:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, TXT allowed")

    DATA_DIR.mkdir(exist_ok=True)
    dest = DATA_DIR / name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = add_file(dest)
    get_pipeline().retriever.refresh()  # BM25 index-ah update pannum
    logger.info("Uploaded %s -> %d chunks", name, chunks)
    return {"filename": name, "chunks": chunks}


@app.get("/sources")
def list_sources():
    return {"sources": get_pipeline().retriever.list_sources()}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        history = [m.model_dump() for m in req.history]
        result = get_pipeline().ask(
            req.question,
            history=history,
            hybrid=req.hybrid,
            rerank=req.rerank,
        )
        logger.info("Q: %s | sources: %d", req.question, len(result["sources"]))
        return result
    except Exception as e:
        logger.exception("ask failed")
        raise HTTPException(status_code=500, detail=str(e))