import sqlite3

import pytest

from src.retrieval.chunker import build_chunks
from src.storage.db import PIPELINE_VERSION, Library
from tests.test_retrieval import make_doc


@pytest.fixture
def lib(tmp_path):
    L = Library(str(tmp_path / "lib.db"))
    yield L
    L.close()


def test_add_is_cached_and_roundtrips(lib):
    a = lib.add_doc_dict("a.pdf", make_doc(), "sha-a")
    b = lib.add_doc_dict("a-again.pdf", make_doc(), "sha-a")
    assert a["cached"] is False and b == {"id": a["id"], "cached": True} and len(lib.list_documents()) == 1
    assert lib.get_doc(a["id"]) == make_doc() and lib.list_documents()[0]["num_pages"] == 4


def test_chunks_index_and_stale_rebuild(lib):
    i = lib.add_doc_dict("a.pdf", make_doc(), "sha-a")["id"]
    ch = lib.get_chunks(i)
    assert ch == build_chunks(make_doc()) and all("sents" in c for c in ch)
    assert lib.get_index(i).query("Is NVDA free?")[0]["page"] == 1
    lib.con.execute("UPDATE documents SET pipeline_version=0")
    lib.con.execute("DELETE FROM chunks")
    assert len(lib.get_chunks(i)) == len(ch)  # rebuilt from the stored doc, no extraction needed
    assert lib.con.execute("SELECT pipeline_version FROM documents").fetchone()[0] == PIPELINE_VERSION


def test_position_and_resume(lib):
    assert lib.resume() is None
    a = lib.add_doc_dict("a.pdf", make_doc(), "sha-a")["id"]
    b = lib.add_doc_dict("b.pdf", make_doc(), "sha-b")["id"]
    lib.save_position(a, 3, "p3_b1")
    lib.touch(a)
    assert lib.resume() == {"doc_id": a, "name": "a.pdf", "page": 3, "block_id": "p3_b1"}
    lib.touch(b)
    assert lib.resume()["doc_id"] == b and lib.resume()["page"] == 1
    assert [d["id"] for d in lib.list_documents()] == [b, a]
    with pytest.raises(ValueError):
        lib.save_position(a, 5)


def test_bookmarks(lib):
    a = lib.add_doc_dict("a.pdf", make_doc(), "sha-a")["id"]
    x, y = lib.add_bookmark(a, 4, "braille"), lib.add_bookmark(a, 2)
    assert [b["page"] for b in lib.list_bookmarks(a)] == [2, 4]
    assert lib.delete_bookmark(x) and not lib.delete_bookmark(x) and len(lib.list_bookmarks(a)) == 1
    with pytest.raises(KeyError):
        lib.add_bookmark(999, 1)
    with pytest.raises(sqlite3.IntegrityError):
        lib.con.execute("INSERT INTO bookmarks(doc_id,page) VALUES(999,1)")


def test_everything_persists_after_reopen(tmp_path):
    path = str(tmp_path / "p.db")
    L = Library(path)
    i = L.add_doc_dict("a.pdf", make_doc(), "sha-a")["id"]
    L.save_position(i, 2)
    L.touch(i)
    L.add_bookmark(i, 3, "contrast")
    L.set_pref("wpm", 150)
    L.set_pref("high_contrast", True)
    L.close()
    L2 = Library(path)
    assert L2.resume()["page"] == 2 and L2.list_bookmarks(i)[0]["label"] == "contrast"
    assert L2.get_pref("wpm") == 150 and L2.get_pref("high_contrast") is True and L2.get_pref("none", "d") == "d"
    assert L2.get_doc(i)["num_pages"] == 4 and L2.add_doc_dict("x", make_doc(), "sha-a")["cached"] is True
    L2.close()


def test_add_document_hashes_file_and_skips_extraction(lib, tmp_path):
    f = tmp_path / "x.pdf"
    f.write_bytes(b"%PDF fake bytes")
    calls = []
    ext = lambda p: calls.append(p) or make_doc()
    r1, r2 = lib.add_document(str(f), extract=ext), lib.add_document(str(f), extract=ext)
    assert r1["cached"] is False and r2 == {"id": r1["id"], "cached": True} and len(calls) == 1
