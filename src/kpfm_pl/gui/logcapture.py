"""Route ``print`` output and warnings into the GUI log panel.

The backend reports progress with ``print`` (e.g. photon counts after loading)
and problems with ``warnings.warn``. :func:`install` redirects both into a
:class:`LogBus`, whose ``message`` signal is safe to emit from worker threads.
"""

from __future__ import annotations

import sys
import threading
import traceback
import warnings

from PySide6.QtCore import QObject
from PySide6.QtCore import Signal as QtSignal

__all__ = ["LogBus", "install"]


class LogBus(QObject):
    """Carries log lines to the GUI; ``level`` is info, warning or error."""

    message = QtSignal(str, str)

    def info(self, text: str) -> None:
        self.message.emit(text, "info")

    def warning(self, text: str) -> None:
        self.message.emit(text, "warning")

    def error(self, text: str) -> None:
        self.message.emit(text, "error")


class _LineWriter:
    """File-like object that emits complete lines and mirrors the original."""

    def __init__(self, bus: LogBus, original) -> None:
        self._bus = bus
        self._original = original
        self._buffer = ""
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        if self._original is not None:
            try:
                self._original.write(text)
            except Exception:
                pass
        with self._lock:
            self._buffer += text
            *lines, self._buffer = self._buffer.split("\n")
        for line in lines:
            if line.strip():
                self._bus.info(line)
        return len(text)

    def flush(self) -> None:
        if self._original is not None:
            try:
                self._original.flush()
            except Exception:
                pass


def install(bus: LogBus):
    """Redirect ``sys.stdout``, warnings and uncaught exceptions to ``bus``.

    Returns a function that undoes the redirection.
    """
    old_stdout = sys.stdout
    old_showwarning = warnings.showwarning
    old_excepthook = sys.excepthook
    sys.stdout = _LineWriter(bus, old_stdout)

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        bus.warning(f"{category.__name__}: {message}")

    def _excepthook(exc_type, exc, tb):
        # Exceptions in Qt callbacks would otherwise vanish when the GUI is
        # started without a console (kpfm-pl-gui on Windows).
        bus.error("Unexpected error: "
                  + "".join(traceback.format_exception_only(exc_type, exc)).strip())
        for line in "".join(traceback.format_tb(tb)).strip().splitlines()[-6:]:
            bus.error("    " + line)
        if sys.__stderr__ is not None:
            old_excepthook(exc_type, exc, tb)

    warnings.showwarning = _showwarning
    sys.excepthook = _excepthook

    def uninstall() -> None:
        sys.stdout = old_stdout
        warnings.showwarning = old_showwarning
        sys.excepthook = old_excepthook

    return uninstall
