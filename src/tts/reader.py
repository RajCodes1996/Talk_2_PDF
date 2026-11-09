def is_heading(block):
    """Delegate to the existing is_heading utility in src.reader.utils."""
    from src.reader.utils import is_heading as _is_heading
    return _is_heading(block)

