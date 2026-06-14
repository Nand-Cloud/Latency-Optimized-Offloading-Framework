"""Small logging helper used by training and experiment drivers."""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "hdrl", level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        root = logging.getLogger("hdrl")
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger("hdrl" if name == "hdrl" else f"hdrl.{name}")
