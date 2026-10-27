"""Embedded matplotlib figure with toolbar and export buttons.

:class:`PlotPanel` is the one plotting widget used by every tab. It owns a
``matplotlib.figure.Figure`` (never a pyplot figure, so no global pyplot state
is touched), adds the standard navigation toolbar and the package's wheel
zoom / middle-drag pan, and offers "Save figure" and "Export data" buttons.

What gets exported is decided by the tab: it registers an ``export_provider``
callable returning an :class:`~kpfm_pl.gui.export.ExportPayload` (or ``None``
while there is nothing to export).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QMessageBox, QPushButton, QVBoxLayout, QWidget

from ..plotting import enable_zoom_pan
from .export import FIGURE_FILTERS, ExportPayload, save_figure, write_payload_csv

__all__ = ["PlotPanel"]

ExportProvider = Callable[[], "ExportPayload | None"]

_HINT = ("Wheel: zoom (Shift: x only, Ctrl: y only) | Middle-drag: pan | "
         "Home button: reset view")


class PlotPanel(QWidget):
    """A matplotlib canvas with navigation toolbar and export buttons.

    Parameters
    ----------
    ctx :
        The application context (used for file dialogs and logging). May be
        ``None`` in tests; the export buttons then do nothing.
    navigation :
        ``"full"`` keeps the toolbar's pan/zoom tools. ``"minimal"`` removes
        them, for plots where the left mouse button draws ROIs (pan/zoom then
        stay available via wheel and middle button).
    """

    def __init__(self, ctx=None, parent: QWidget | None = None, *,
                 navigation: str = "full", figsize=(8, 5)) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.figure = Figure(figsize=figsize, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        # Keyboard focus is needed for the selectors' Delete key.
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        if navigation == "minimal":
            for action in self.toolbar.actions():
                if action.text() in ("Pan", "Zoom", "Subplots", "Customize"):
                    self.toolbar.removeAction(action)
        self.toolbar.setToolTip(_HINT)
        enable_zoom_pan(self.figure)
        self.export_provider: ExportProvider | None = None

        self.save_button = QPushButton("Save figure…")
        self.save_button.setToolTip("Save this plot as PNG, SVG or PDF. The analysis "
                                    "parameters are stored in the file's metadata.")
        self.csv_button = QPushButton("Export data (CSV)…")
        self.csv_button.setToolTip("Save the plotted data as a CSV file. Analysis "
                                   "parameters are written as '#' comment lines on top.")
        self.save_button.clicked.connect(self.save_figure_dialog)
        self.csv_button.clicked.connect(self.export_csv_dialog)

        top = QHBoxLayout()
        top.addWidget(self.toolbar, 1)
        top.addWidget(self.save_button)
        top.addWidget(self.csv_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addWidget(self.canvas, 1)
        self.update_export_buttons()

    # ----------------------------------------------------------------- drawing
    def clear(self) -> None:
        """Remove all axes (and colorbars) from the figure."""
        self.figure.clear()

    def draw(self) -> None:
        """Redraw the canvas and reset the toolbar's Home view."""
        self.canvas.draw_idle()
        self.toolbar.update()
        self.update_export_buttons()

    def show_message(self, text: str) -> None:
        """Replace the plot with a centred hint (e.g. "Load a file first")."""
        self.clear()
        self.figure.text(0.5, 0.5, text, ha="center", va="center", color="0.4",
                         wrap=True)
        self.draw()

    # ------------------------------------------------------------------ export
    def set_export_provider(self, provider: ExportProvider | None) -> None:
        self.export_provider = provider
        self.update_export_buttons()

    def update_export_buttons(self) -> None:
        has_axes = bool(self.figure.axes)
        self.save_button.setEnabled(has_axes)
        self.csv_button.setEnabled(has_axes and self.export_provider is not None)

    def _payload(self) -> ExportPayload | None:
        if self.export_provider is None:
            return None
        return self.export_provider()

    def save_figure_dialog(self) -> None:
        if self.ctx is None:
            return
        payload = self._payload()
        name = payload.default_name if payload else "figure"
        metadata = payload.metadata if payload else {"analysis": "kpfm_pl figure"}
        path = self.ctx.save_file_dialog("Save figure", f"{name}.png", FIGURE_FILTERS)
        if not path:
            return
        path = Path(path)
        if path.suffix.lower() not in (".png", ".svg", ".pdf"):
            path = path.with_suffix(".png")
        try:
            save_figure(self.figure, path, metadata)
        except Exception as exc:
            QMessageBox.warning(self, "Save figure", f"Could not save the figure:\n{exc}")
            return
        self.ctx.log.info(f"Figure saved: {path}")

    def export_csv_dialog(self) -> None:
        if self.ctx is None:
            return
        payload = self._payload()
        if payload is None:
            QMessageBox.information(self, "Export data", "There is no data to export yet.")
            return
        path = self.ctx.save_file_dialog("Export data", f"{payload.default_name}.csv",
                                         "CSV file (*.csv)")
        if not path:
            return
        path = Path(path)
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        try:
            write_payload_csv(path, payload)
        except Exception as exc:
            QMessageBox.warning(self, "Export data", f"Could not export the data:\n{exc}")
            return
        self.ctx.log.info(f"Data exported: {path}")
