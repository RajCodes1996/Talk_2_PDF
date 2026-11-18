# src/api/app.py
"""FastAPI entry‑point for Talk_2_PDF.

Endpoints:
* POST   /upload                     – upload a PDF, store it, and return a doc_id.
* GET    /document/{doc_id}/position – current reading position.
* POST   /document/{doc_id}/position – update reading position (page, block_id).
* GET    /document/{doc_id}/bookmarks – list all bookmarks for the document.
* POST   /document/{doc_id}/bookmarks – add a new bookmark.
* POST   /question                    – ask a question on a document, get answer text.
* GET    /health                      – simple health‑check.
"""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse

from src.storage.db import Library
# Heavy imports are loaded lazily inside the request handlers to speed up startup.

# Lazy import: Index is imported inside request handlers to speed up startup.
# Heavy imports are loaded lazily inside request handlers to speed up startup.
# # from src.nlp.answer import Answerer  # lazy import inside ask_question  # moved inside ask_question


from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path

app = FastAPI()   # must come BEFORE app.mount

# Mount static files (CSS/JS)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Simple UI endpoint
@app.get("/", response_class=HTMLResponse)
def ui():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return html_path.read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# Helper – singleton Library (in‑memory for this process)
# ----------------------------------------------------------------------
LIB = Library()  # uses default data/library.db

# ----------------------------------------------------------------------
# Upload / add document
# ----------------------------------------------------------------------
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Store the uploaded PDF, run extraction/OCR, and add it to the Library.
    Returns the newly created *doc_id*.
    """
    dest = UPLOAD_DIR / Path(file.filename).name
    with dest.open("wb") as f:
        while content := await file.read(1024 * 1024):
            f.write(content)
    result = LIB.add_document(str(dest), classify=False)
    LIB.touch(result["id"])
    return {"doc_id": result["id"], "cached": result["cached"]}

# ----------------------------------------------------------------------
# Reading position
# ----------------------------------------------------------------------
@app.get("/document/{doc_id}/position")
def get_position(doc_id: int):
    try:
        pos = LIB.get_position(doc_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    return pos


@app.post("/document/{doc_id}/position")
def set_position(
    doc_id: int,
    page: int = Body(..., embed=True),
    block_id: Optional[str] = Body(None, embed=True),
):
    try:
        LIB.save_position(doc_id, page, block_id)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    return {"status": "saved", "doc_id": doc_id, "page": page, "block_id": block_id}

# ----------------------------------------------------------------------
# Bookmarks
# ----------------------------------------------------------------------
@app.get("/document/{doc_id}/bookmarks")
def list_bookmarks(doc_id: int):
    return LIB.list_bookmarks(doc_id)


@app.post("/document/{doc_id}/bookmarks")
def add_bookmark(
    doc_id: int,
    page: int = Body(..., embed=True),
    label: str = Body("", embed=True),
    block_id: Optional[str] = Body(None, embed=True),
):
    try:
        bm_id = LIB.add_bookmark(doc_id, page, label, block_id)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    return {"bookmark_id": bm_id, "doc_id": doc_id, "page": page, "label": label}

# ----------------------------------------------------------------------
# Question answering
# ----------------------------------------------------------------------
from pydantic import BaseModel

class QARequest(BaseModel):
    doc_id: int
    query: str

@app.post("/question")
def ask_question(payload: QARequest):
    # Verify document exists – give a clear 404 instead of a generic 500
    if not LIB.con.execute("SELECT 1 FROM documents WHERE id=?", (payload.doc_id,)).fetchone():
        raise HTTPException(status_code=404, detail="Document not found")
    """Return the best answer (sentence(s)) for *payload.query* on *payload.doc_id*.
    """
    from src.nlp.answer import Answerer

    idx = LIB.get_index(payload.doc_id)
    answerer = Answerer(idx)
    answer = answerer.answer(payload.query)
    return {"answer": answer, "doc_id": payload.doc_id}

# ----------------------------------------------------------------------
# Health‑check
# ----------------------------------------------------------------------
@app.get("/health")
def health():
    return JSONResponse(content={"status": "ok"})
