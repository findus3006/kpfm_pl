"""Graphical user interface for kpfm_pl (optional).

Requires the ``gui`` extra (``pip install -e .[gui]``: PySide6 and matplotlib).
Start it with::

    kpfm-pl-gui
    # or
    python -m kpfm_pl.gui

The GUI is a thin front end: it only uses the public ``kpfm_pl`` API, and
nothing in the core package imports it, so the backend stays usable headless
and independent of Qt. The Qt-free workflow logic lives in
:mod:`kpfm_pl.gui.models`, the widgets in :mod:`kpfm_pl.gui.tabs`.
"""

from __future__ import annotations

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Launch the kpfm_pl GUI and return its exit code."""
    try:
        from .app import run
    except ImportError as exc:  # PySide6 or matplotlib missing
        raise SystemExit(
            f"The kpfm_pl GUI needs PySide6 and matplotlib ({exc}).\n"
            "Install them with:  pip install -e .[gui]"
        ) from exc
    return run(argv)
