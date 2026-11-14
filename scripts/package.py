"""Package the application for offline use.

This script assumes `pyinstaller` is available in the current environment.
It builds a single‑file executable that bundles the source code, the
pre‑trained classifier models (in `data/models/`), and the TTS engine.

Usage::
    python scripts/package.py

The resulting executable is placed in `dist/vision.exe` (Windows) or
`dist/vision` (Unix).  No network access is required to run the produced
binary.
"""
import subprocess
import sys
import os

def main():
    # Ensure we are in the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(project_root)

    # PyInstaller command – one‑file, console disabled, add data files
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", "vision",
        "--add-data", "data/models;data/models",
        "--add-data", "data/samples;data/samples",
        "--hidden-import", "src",
        "vision/__main__.py" if os.path.exists("vision/__main__.py") else "src/__main__.py",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("Packaging complete. Executable is in ./dist")

if __name__ == "__main__":
    main()
