"""Loaded measurement files shared between the GUI tabs (no Qt).

A :class:`LoadedFile` wraps one backend reader (``PTU``, ``Zurich`` or
``JPK``) together with a re-entrant lock. Readers keep internal state between
calls (a ``PTU`` caches its most recent time trace), so background jobs must
hold ``handle.lock`` for the whole sequence of calls that belongs together.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...jpk import JPK
from ...ptu import PTU
from ...zurich import Zurich

__all__ = ["LoadedFile", "load_ptu", "load_zurich", "load_jpk", "KINDS"]

#: File kinds the GUI understands, with their dialog filters.
KINDS = {
    "ptu": "PicoQuant PTU (*.ptu)",
    "zurich": "Zurich lock-in export (*.txt)",
    "jpk": "JPK AFM image (*.jpk)",
}


@dataclass(eq=False)
class LoadedFile:
    """A loaded reader plus the lock that serialises access to it."""

    kind: str
    path: Path
    reader: Any
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem


def load_ptu(path: str | Path, dark_counts: float = 0.0) -> LoadedFile:
    """Read a ``.ptu`` file (slow for large files: run it in a job)."""
    path = Path(path)
    return LoadedFile("ptu", path, PTU(path, dark_counts=dark_counts))


def load_zurich(path: str | Path) -> LoadedFile:
    """Read a Zurich ``.txt`` export."""
    path = Path(path)
    return LoadedFile("zurich", path, Zurich(path))


def load_jpk(path: str | Path) -> LoadedFile:
    """Read a JPK ``.jpk`` image file."""
    path = Path(path)
    return LoadedFile("jpk", path, JPK(path))
