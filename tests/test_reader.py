import subprocess
import sys
from pathlib import Path

import pytest

# Path to a sample processed JSON file
SAMPLE_JSON = Path(__file__).parents[1] / "data" / "processed" / "sam.json"

@pytest.mark.skipif(not SAMPLE_JSON.exists(), reason="Sample JSON not found")
def test_reader_cli_list_headings():
    """Run the reader CLI with list-headings and ensure it exits cleanly and prints output."""
    cmd = [sys.executable, "-m", "src.reader", str(SAMPLE_JSON), "--command", "list-headings"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # The command should exit with code 0
    assert result.returncode == 0, f"Non‑zero exit: {result.returncode}\nStderr: {result.stderr}"
    # Output may be empty if no headings, but command should run without error
    # Ensure no error text is printed
    assert "Error" not in result.stderr

def test_reader_class_navigation():
    """Import the Reader class and exercise navigation methods without audio output."""
    from src.reader.reader import Reader
    reader = Reader(str(SAMPLE_JSON))
    # Initial block should be 0,0
    assert reader.page_idx == 0
    assert reader.block_idx == 0
    # Move to next block and back
    reader.next_block()
    assert reader.block_idx == 1
    reader.prev_block()
    assert reader.block_idx == 0
    # Page navigation (if more than one page)
    if len(reader.pages) > 1:
        cur_page = reader.page_idx
        reader.next_page()
        assert reader.page_idx == cur_page + 1
        reader.prev_page()
        assert reader.page_idx == cur_page
    # List headings should return a list (could be empty)
    headings = reader.list_headings()
    assert isinstance(headings, list)

