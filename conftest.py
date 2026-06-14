"""Make the repo root importable so tests can ``import src...``."""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
# Make both ``<repo>/src`` (compat package) and ``<repo>/src/src`` (actual code)
# discoverable for imports like ``import src.common``.
SRC_ROOT = os.path.join(REPO_ROOT, "src")
INNER_SRC = os.path.join(SRC_ROOT, "src")
for p in (SRC_ROOT, INNER_SRC, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
