def test_imports():
    import pymupdf, sklearn, rank_bm25, pyttsx3, pytesseract, cv2


def test_tts_engine():
    import pyttsx3
    assert pyttsx3.init() is not None
