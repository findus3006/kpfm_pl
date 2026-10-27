"""ROI editing for the time-trace plot: box selector, ROI list and number entry.

:class:`RoiEditor` keeps the list of ROIs (``{x_min, x_max, y_min, y_max}``
dicts), shows it in a list widget, and mirrors it on an
:class:`~kpfm_pl.selector.InteractiveBoxSelector` attached to the trace axes,
so boxes can be drawn, moved, resized and deleted with the mouse.
"""

from __future__ import annotations

from matplotlib.colors import to_hex
from PySide6.QtCore import QObject
from PySide6.QtCore import Signal as QtSignal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHBoxLayout, QListWidget, QListWidgetItem, QPushButton

from ...selector import InteractiveBoxSelector
from ..forms import ParamForm

__all__ = ["RoiEditor", "roi_colour"]


def roi_colour(index: int) -> str:
    """Matplotlib cycle colour of ROI ``index`` (shared by list, trace and results)."""
    return f"C{index % 10}"


class RoiEditor(QObject):
    """ROI list + selector for one :class:`~kpfm_pl.gui.canvas.PlotPanel`.

    Builds its widgets into ``layout`` and emits ``changed`` whenever the
    ROIs change (drawn, moved, removed, typed in or replaced).
    """

    changed = QtSignal()

    def __init__(self, ctx, panel, layout, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.panel = panel
        self.rois: list[dict[str, float]] = []
        self.selector: InteractiveBoxSelector | None = None
        self._labels = []
        self._event_cids: list[int] = []

        self.list = QListWidget()
        self.list.setMinimumHeight(90)
        self.list.setToolTip("ROIs drawn on the time trace: time window and range of "
                             "counts per bin. Colours match the fits and FLID maps.")
        layout.addWidget(self.list)
        row = QHBoxLayout()
        self.remove_button = QPushButton("Remove selected")
        self.remove_button.setToolTip("Delete the ROI selected in the list.")
        self.remove_button.clicked.connect(self.remove_selected)
        self.clear_button = QPushButton("Clear all")
        self.clear_button.setToolTip("Delete all ROIs.")
        self.clear_button.clicked.connect(self.clear)
        row.addWidget(self.remove_button)
        row.addWidget(self.clear_button)
        layout.addLayout(row)

        f = self.form = ParamForm()
        f.add_float("x_min", "From", 0.0, maximum=1e6, step=1, decimals=2, suffix=" s",
                    tooltip="Start of the ROI time window.")
        f.add_float("x_max", "To", 10.0, maximum=1e6, step=1, decimals=2, suffix=" s",
                    tooltip="End of the ROI time window.")
        f.add_float("y_min", "Counts from", 0.0, maximum=1e9, step=10, decimals=1,
                    tooltip="Lowest counts per bin included in the ROI.")
        f.add_float("y_max", "Counts to", 1000.0, maximum=1e9, step=10, decimals=1,
                    tooltip="Highest counts per bin included in the ROI.")
        layout.addWidget(f)
        self.add_button = QPushButton("Add ROI from numbers")
        self.add_button.setToolTip("Add an ROI with exactly the values above instead of "
                                   "drawing it.")
        self.add_button.clicked.connect(self.add_from_form)
        layout.addWidget(self.add_button)

    # ---------------------------------------------------------------- editing
    def set_rois(self, rois, *, notify: bool = True) -> None:
        """Replace all ROIs (list of ``{x_min, x_max, y_min, y_max}`` dicts)."""
        self.rois = [dict(r) for r in rois]
        if self.selector is not None:
            self.selector.set_boxes(self.rois)
        self._refresh_list()
        self._decorate()
        if notify:
            self.changed.emit()

    def add(self, roi: dict[str, float]) -> None:
        self.set_rois([*self.rois, roi])

    def add_from_form(self) -> None:
        v = self.form.values()
        if v["x_max"] <= v["x_min"] or v["y_max"] <= v["y_min"]:
            self.ctx.log.warning("ROI not added: 'To' must be larger than 'From'.")
            return
        self.add(v)

    def remove_selected(self) -> None:
        row = self.list.currentRow()
        if 0 <= row < len(self.rois):
            self.set_rois(self.rois[:row] + self.rois[row + 1:])

    def clear(self) -> None:
        self.set_rois([])

    # ------------------------------------------------------------- selector
    def attach(self, ax) -> None:
        """(Re)attach the box selector to freshly drawn trace axes ``ax``."""
        self.detach()
        self.selector = InteractiveBoxSelector(ax)
        # Matplotlib calls handlers in connection order, so connecting after the
        # selector lets these read its boxes once it has processed the event.
        canvas = self.panel.canvas
        self._event_cids = [
            canvas.mpl_connect("button_release_event", lambda _e: self._sync()),
            canvas.mpl_connect("key_press_event", lambda _e: self._sync()),
        ]
        self.selector.set_boxes(self.rois)
        self._decorate()

    def detach(self) -> None:
        """Disconnect the selector (e.g. before the axes are cleared)."""
        if self.selector is not None:
            self.selector.disconnect()
            self.selector = None
        for cid in self._event_cids:
            self.panel.canvas.mpl_disconnect(cid)
        self._event_cids = []
        self._labels = []

    def _sync(self) -> None:
        """Copy the selector's boxes after a mouse/keyboard interaction."""
        if self.selector is None:
            return
        boxes = self.selector.get_boxes()
        if boxes != self.rois:
            self.rois = boxes
            self._refresh_list()
            self._decorate()
            self.changed.emit()
        else:
            self._decorate()

    def _refresh_list(self) -> None:
        self.list.clear()
        for i, roi in enumerate(self.rois):
            item = QListWidgetItem(
                f"ROI {i + 1}: {roi['x_min']:.1f}–{roi['x_max']:.1f} s, "
                f"{roi['y_min']:.0f}–{roi['y_max']:.0f} counts")
            item.setForeground(QColor(to_hex(roi_colour(i))))
            self.list.addItem(item)

    def _decorate(self) -> None:
        """Colour the boxes like their results and number them."""
        for text in self._labels:
            try:
                text.remove()
            except (ValueError, NotImplementedError):  # axes already cleared
                pass
        self._labels = []
        if self.selector is None:
            return
        ax = self.selector.ax
        for i, (box, roi) in enumerate(zip(self.selector.boxes, self.rois)):
            if box.get("patch") is not None:
                box["patch"].set_facecolor(roi_colour(i))
            self._labels.append(ax.text(
                roi["x_min"], roi["y_max"], f" {i + 1}", color=roi_colour(i),
                fontweight="bold", va="top", ha="left", zorder=6, clip_on=True))
        self.panel.canvas.draw_idle()
