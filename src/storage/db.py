"""SQLite library: documents, cached chunks, reading position, bookmarks and preferences. Stdlib only, offline."""
import hashlib
import json
import os
import sqlite3
import threading
import zlib

from src.retrieval.chunker import build_chunks
from src.retrieval.index import Index

PIPELINE_VERSION = 1  # bump when extractor/chunker output changes: cached chunks are rebuilt from the stored doc

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, path TEXT, sha256 TEXT UNIQUE NOT NULL, num_pages INTEGER NOT NULL,
  doc BLOB NOT NULL, pipeline_version INTEGER NOT NULL, last_opened INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS chunks(
  doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE, idx INTEGER NOT NULL, data TEXT NOT NULL,
  PRIMARY KEY(doc_id, idx));
CREATE TABLE IF NOT EXISTS position(
  doc_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE, page INTEGER NOT NULL, block_id TEXT);
CREATE TABLE IF NOT EXISTS bookmarks(
  id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  page INTEGER NOT NULL, block_id TEXT, label TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS prefs(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class Library:
    def __init__(self, path="data/library.db"):
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA foreign_keys=ON")
        self.con.executescript(SCHEMA)
        self.lock = threading.RLock()
        self._docs, self._idx = {}, {}

    def close(self):
        self.con.close()

    # ---- documents
    def add_doc_dict(self, name, doc, sha256, path=None):
        """Store a doc (extractor JSON dict). Returns {'id', 'cached'}; same sha256 = cache hit, no recompute."""
        with self.lock:
            row = self.con.execute("SELECT id FROM documents WHERE sha256=?", (sha256,)).fetchone()
            if row:
                return {"id": row["id"], "cached": True}
            blob = zlib.compress(json.dumps(doc, ensure_ascii=False).encode("utf-8"), 6)
            n = doc.get("num_pages") or len(doc["pages"])
            with self.con:
                cur = self.con.execute(
                    "INSERT INTO documents(name,path,sha256,num_pages,doc,pipeline_version) VALUES(?,?,?,?,?,?)",
                    (name, path, sha256, n, blob, PIPELINE_VERSION))
                self._store_chunks(cur.lastrowid, build_chunks(doc))
            return {"id": cur.lastrowid, "cached": False}

    def add_document(self, pdf_path, classify=False, extract=None):
        """Hash the file first: an already‑stored file is returned without re‑running extraction/OCR."""
        h = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        sha = h.hexdigest()
        row = self.con.execute("SELECT id FROM documents WHERE sha256=?", (sha,)).fetchone()
        if row:
            return {"id": row["id"], "cached": True}
        if extract is None:
            from src.ingestion.extractor import extract_document as extract
        doc = extract(pdf_path)
        if classify:
            from pathlib import Path
            from src.nlp.block_classifier import MODEL_PATH, classify_document
            if Path(MODEL_PATH).exists():
                classify_document(doc)
        return self.add_doc_dict(os.path.basename(pdf_path), doc, sha, os.path.abspath(pdf_path))

    def _store_chunks(self, doc_id, chunks):
        self.con.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        self.con.executemany(
            "INSERT INTO chunks(doc_id,idx,data) VALUES(?,?,?)",
            [(doc_id, i, json.dumps(c, ensure_ascii=False)) for i, c in enumerate(chunks)])

    def get_doc(self, doc_id):
        if doc_id not in self._docs:
            row = self.con.execute("SELECT doc FROM documents WHERE id=?", (doc_id,)).fetchone()
            if not row:
                raise KeyError(f"no document {doc_id}")
            self._docs[doc_id] = json.loads(zlib.decompress(row["doc"]))
        return self._docs[doc_id]

    def get_chunks(self, doc_id):
        with self.lock:
            row = self.con.execute("SELECT pipeline_version FROM documents WHERE id=?", (doc_id,)).fetchone()
            if not row:
                raise KeyError(f"no document {doc_id}")
            if row["pipeline_version"] != PIPELINE_VERSION:
                with self.con:
                    self._store_chunks(doc_id, build_chunks(self.get_doc(doc_id)))
                    self.con.execute("UPDATE documents SET pipeline_version=? WHERE id=?", (PIPELINE_VERSION, doc_id))
                self._idx.pop(doc_id, None)
            rows = self.con.execute("SELECT data FROM chunks WHERE doc_id=? ORDER BY idx", (doc_id,)).fetchall()
            return [json.loads(r["data"]) for r in rows]

    def get_index(self, doc_id):
        """Search index rebuilt from cached chunks (fast; no extraction)."""
        if doc_id not in self._idx or self._stale(doc_id):
            self._idx[doc_id] = Index(self.get_chunks(doc_id))
        return self._idx[doc_id]

    def _stale(self, doc_id):
        r = self.con.execute("SELECT pipeline_version FROM documents WHERE id=?", (doc_id,)).fetchone()
        return bool(r) and r["pipeline_version"] != PIPELINE_VERSION

    def list_documents(self):
        rows = self.con.execute(
            "SELECT d.id,d.name,d.num_pages,d.last_opened,p.page FROM documents d LEFT JOIN position p ON p.doc_id=d.id "
            "ORDER BY d.last_opened DESC, d.id DESC").fetchall()
        return [{"id": r["id"], "name": r["name"], "num_pages": r["num_pages"], "last_opened": r["last_opened"],
                 "page": r["page"] or 1} for r in rows]

    # ---- reading position
    def touch(self, doc_id):
        """Mark a document as the most recently opened (monotonic counter, safe on coarse Windows clocks)."""
        with self.lock, self.con:
            self.con.execute(
                "UPDATE documents SET last_opened=(SELECT COALESCE(MAX(last_opened),0)+1 FROM documents) WHERE id=?",
                (doc_id,))

    def save_position(self, doc_id, page, block_id=None):
        n = self._pages(doc_id)
        if not 1 <= page <= n:
            raise ValueError(f"page {page} outside 1..{n}")
        with self.lock, self.con:
            self.con.execute(
                "INSERT INTO position(doc_id,page,block_id) VALUES(?,?,?) ON CONFLICT(doc_id) DO UPDATE SET page=excluded.page, block_id=excluded.block_id",
                (doc_id, page, block_id))

    def get_position(self, doc_id):
        r = self.con.execute("SELECT page,block_id FROM position WHERE doc_id=?", (doc_id,)).fetchone()
        return {"page": r["page"], "block_id": r["block_id"]} if r else {"page": 1, "block_id": None}

    def resume(self):
        r = self.con.execute("SELECT id,name FROM documents WHERE last_opened>0 ORDER BY last_opened DESC LIMIT 1").fetchone()
        return None if not r else {"doc_id": r["id"], "name": r["name"], **self.get_position(r["id"])}

    def _pages(self, doc_id):
        r = self.con.execute("SELECT num_pages FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not r:
            raise KeyError(f"no document {doc_id}")
        return r["num_pages"]

    # ---- bookmarks
    def add_bookmark(self, doc_id, page, label="", block_id=None):
        if not 1 <= page <= self._pages(doc_id):
            raise ValueError("page out of range")
        with self.lock, self.con:
            return self.con.execute(
                "INSERT INTO bookmarks(doc_id,page,block_id,label) VALUES(?,?,?,?)",
                (doc_id, page, block_id, label)).lastrowid

    def list_bookmarks(self, doc_id):
        rows = self.con.execute(
            "SELECT id,page,block_id,label FROM bookmarks WHERE doc_id=? ORDER BY page,id", (doc_id,))
        return [dict(r) for r in rows]

    def delete_bookmark(self, bookmark_id):
        with self.lock, self.con:
            return self.con.execute("DELETE FROM bookmarks WHERE id=?", (bookmark_id,)).rowcount == 1

    # ---- preferences
    def set_pref(self, key, value):
        with self.lock, self.con:
            self.con.execute(
                "INSERT INTO prefs(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)))

    def get_pref(self, key, default=None):
        r = self.con.execute("SELECT value FROM prefs WHERE key=?", (key,)).fetchone()
        return json.loads(r["value"]) if r else default

    def all_prefs(self):
        return {r["key"]: json.loads(r["value"]) for r in self.con.execute("SELECT key,value FROM prefs ORDER BY key")}

