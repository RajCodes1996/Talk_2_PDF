from __future__ import annotations

import json
from pathlib import Path
import pyttsx3


class Reader:
    """Navigate and read extracted PDF blocks aloud.

    The JSON format follows the schema from `src.ingestion.extractor` and includes
    new fields `source`, `ocr_conf`, `low_confidence`, `ocr_applied`, `skew_angle`.
    """

    def __init__(self, json_path: str | Path):
        self.path = Path(json_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Result file not found: {self.path}")
        with self.path.open(encoding="utf-8") as f:
            self.doc = json.load(f)
        self.pages = self.doc.get("pages", [])
        self.page_idx = 0
        self.block_idx = 0
        self.engine = pyttsx3.init()
        # default speech properties
        self.engine.setProperty("rate", 150)
        self.engine.setProperty("volume", 1.0)

    # ---------------------------------------------------------------------
    # Navigation helpers
    # ---------------------------------------------------------------------
    def _current_page(self) -> dict:
        return self.pages[self.page_idx]

    def _current_block(self) -> dict:
        blocks = self._current_page().get("blocks", [])
        return blocks[self.block_idx]

    def next_block(self) -> dict:
        blocks = self._current_page().get("blocks", [])
        if self.block_idx + 1 < len(blocks):
            self.block_idx += 1
        else:
            # move to next page if possible
            if self.page_idx + 1 < len(self.pages):
                self.page_idx += 1
                self.block_idx = 0
        return self._current_block()

    def prev_block(self) -> dict:
        if self.block_idx > 0:
            self.block_idx -= 1
        else:
            if self.page_idx > 0:
                self.page_idx -= 1
                self.block_idx = max(0, len(self._current_page().get("blocks", [])) - 1)
        return self._current_block()

    def next_page(self) -> dict:
        if self.page_idx + 1 < len(self.pages):
            self.page_idx += 1
            self.block_idx = 0
        return self._current_page()

    def prev_page(self) -> dict:
        if self.page_idx > 0:
            self.page_idx -= 1
            self.block_idx = 0
        return self._current_page()

    def jump_to_heading(self, text: str) -> dict:
        """Jump to the first block whose text contains *text* and looks like a heading.
        A heading is identified by a high `size_ratio` (>1.4) and `bold_ratio` == 1.0.
        """
        for pi, page in enumerate(self.pages):
            for bi, block in enumerate(page.get("blocks", [])):
                if (
                    text.lower() in block.get("text", "").lower()
                    and block.get("size_ratio", 0) > 1.4
                    and block.get("bold_ratio", 0) == 1.0
                ):
                    self.page_idx = pi
                    self.block_idx = bi
                    return block
        raise ValueError(f"Heading containing '{text}' not found")

    # ---------------------------------------------------------------------
    # Speech control
    # ---------------------------------------------------------------------
    def set_speed(self, wpm: int) -> None:
        self.engine.setProperty("rate", wpm)

    def set_voice(self, voice_name: str) -> None:
        voices = self.engine.getProperty("voices")
        for v in voices:
            if voice_name.lower() in v.name.lower():
                self.engine.setProperty("voice", v.id)
                return
        raise ValueError(f"Voice '{voice_name}' not found")

    def speak_current(self, announce_page: bool = True) -> None:
        block = self._current_block()
        text = block.get("text", "")
        if announce_page:
            prefix = f"Page {self.page_idx + 1}, block {self.block_idx + 1}: "
        else:
            prefix = ""
        self.engine.say(prefix + text)
        self.engine.runAndWait()

    # ---------------------------------------------------------------------
    # Utility
    # ---------------------------------------------------------------------
    def list_headings(self) -> list[tuple[int, int, str]]:
        """Return a list of (page_index, block_index, text) for heading‑like blocks."""
        headings = []
        for pi, page in enumerate(self.pages):
            for bi, block in enumerate(page.get("blocks", [])):
                if block.get("size_ratio", 0) > 1.4 and block.get("bold_ratio", 0) == 1.0:
                    headings.append((pi, bi, block.get("text", "")))
        return headings

