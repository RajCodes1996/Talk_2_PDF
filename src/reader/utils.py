"""Utility helpers for the Reader module."""

def is_heading(block: dict) -> bool:
    """Return True if a block looks like a heading.
    Heuristic: size_ratio > 1.4 and bold_ratio == 1.0.
    """
    return block.get("size_ratio", 0) > 1.4 and block.get("bold_ratio", 0) == 1.0

