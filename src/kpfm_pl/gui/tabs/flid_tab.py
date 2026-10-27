"""FLID tab: draw regions on the time trace and compare their FLID maps."""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Rectangle
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTabWidget,
)

from ...selector import InteractiveBoxSelector
from ..canvas import PlotPanel
from ..forms import ParamForm
from ..models import flid as model
from ..models.ptu import FIT_MODES, TraceParams, compute_trace
from .base import BaseTab

__all__ = ["FlidTab"]


def _colour(index: int) -> str:
    """Matplotlib cycle colour of ROI ``index`` (shared by list, trace and maps)."""
    return f"C{index % 10}"


class FlidTab(BaseTab):
    """Draw ROI boxes on a PTU time trace and compute one FLID per box."""

    title = "FLID"

    def __init__(self, ctx, parent=None) -> None:
        super().__init__(ctx, parent)
        self.trace = None                 # models.ptu.TraceResult of the drawn trace
        self.result: model.FlidResult | None = None
        self.rois: list[dict[str, float]] = []
        self.selector: InteractiveBoxSelector | None = None
        self._roi_labels = []
        self._event_cids: list[int] = []

        self.add_hint(
            "1. Open a .ptu file and check the bin time of the time trace.\n"
            "2. Draw regions (ROIs) on the trace: left-drag draws a box, drag inside "
            "moves it, drag a handle resizes it, Delete removes the selected box.\n"
            "3. Press 'Compute FLIDs'. Without ROIs the window between the first and "
            "last laser trigger is used.\n"
            "Hover over a field for an explanation."
        )
        self._build_file_group()
        self._build_trace_group()
        self._build_roi_group()
        self._build_flid_group()
        self._build_export_group()

        self.plots = QTabWidget()
        self.trace_plot = PlotPanel(ctx, navigation="minimal")
        self.flid_plot = PlotPanel(ctx)
        self.plots.addTab(self.trace_plot, "Time trace && ROIs")
        self.plots.addTab(self.flid_plot, "FLID maps")
        self.content_layout.addWidget(self.plots)
        self.trace_plot.set_export_provider(self._trace_payload)
        self.flid_plot.set_export_provider(self._flid_payload)
        self.trace_plot.show_message("Open a .ptu file to start.")
        self.flid_plot.show_message("Draw ROIs on the time trace and press 'Compute FLIDs'.")
        self._set_enabled(False)

    # ------------------------------------------------------------- controls
    def _build_file_group(self) -> None:
        layout = self.add_group("File")
        self.file_label = QLabel("No PTU file loaded.")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)
        self.add_button(layout, "Open PTU file…", lambda: self.ctx.open_and_load("ptu"))

    def _build_trace_group(self) -> None:
        layout = self.add_group("Time trace")
        f = self.trace_form = ParamForm()
        f.add_float("bintime", "Bin time", 0.5, minimum=1e-3, maximum=600, step=0.1,
                    decimals=3, suffix=" s",
                    tooltip="Width of one time bin. The FLID fits every bin separately "
                            "and uses the same bin time, so the ROI count ranges keep "
                            "their meaning. Changing it rescales existing ROIs.")
        f.add_float("dark_counts", "Dark counts", 0.0, maximum=1e9, step=0.25, decimals=3,
                    suffix=" /s", tooltip="Detector dark counts per second, subtracted as "
                                          "a constant offset in the per-bin fits.")
        f.add_bool("laser_pulse", "Laser markers", True,
                   text="recorded in marker channel",
                   tooltip="Checked (default, current recordings): laser triggers are read "
                           "from the PTU marker channel.\nUnchecked: legacy files without "
                           "markers; the non-zero part of the trace is centred.")
        f.add_int("marker_channel", "Marker channel", 4, minimum=1, maximum=15,
                  tooltip="PicoHarp marker channel carrying the laser trigger.")
        layout.addWidget(f)
        self.trace_button = self.add_button(layout, "Update time trace", self.update_trace,
                                            primary=True)

    def _build_roi_group(self) -> None:
        layout = self.add_group("Regions of interest")
        self.roi_list = QListWidget()
        self.roi_list.setMinimumHeight(110)
        self.roi_list.setToolTip("ROIs drawn on the time trace: time window and range of "
                                 "counts per bin. Colours match the FLID maps.")
        layout.addWidget(self.roi_list)
        row = QHBoxLayout()
        self.remove_button = QPushButton("Remove selected")
        self.remove_button.setToolTip("Delete the ROI selected in the list.")
        self.remove_button.clicked.connect(self.remove_selected_roi)
        self.clear_button = QPushButton("Clear all")
        self.clear_button.setToolTip("Delete all ROIs.")
        self.clear_button.clicked.connect(self.clear_rois)
        row.addWidget(self.remove_button)
        row.addWidget(self.clear_button)
        layout.addLayout(row)

        f = self.roi_form = ParamForm()
        f.add_float("x_min", "From", 0.0, maximum=1e6, step=1, decimals=2, suffix=" s",
                    tooltip="Start of the ROI time window.")
        f.add_float("x_max", "To", 10.0, maximum=1e6, step=1, decimals=2, suffix=" s",
                    tooltip="End of the ROI time window.")
        f.add_float("y_min", "Counts from", 0.0, maximum=1e9, step=10, decimals=1,
                    tooltip="Lowest counts per bin included in the ROI.")
        f.add_float("y_max", "Counts to", 1000.0, maximum=1e9, step=10, decimals=1,
                    tooltip="Highest counts per bin included in the ROI.")
        layout.addWidget(f)
        self.add_roi_button = self.add_button(
            layout, "Add ROI from numbers", self.add_roi_from_form,
            tooltip="Add an ROI with exactly the values above instead of drawing it.")

    def _build_flid_group(self) -> None:
        layout = self.add_group("FLID")
        f = self.flid_form = ParamForm()
        f.add_int("n_exp", "Exponentials", 1, minimum=1, maximum=4,
                  tooltip="Exponential components fitted per time bin. With more than "
                          "one, the map uses their intensity-weighted mean lifetime.")
        f.add_int("bins", "Lifetime binning", 1, minimum=1, maximum=64,
                  tooltip="Merge this many lifetime channels before fitting "
                          "(helps with few photons per bin).")
        f.add_float("lifetime_min", "Lifetime from", 0.0, maximum=1000, step=1, decimals=2,
                    suffix=" ns", tooltip="Lower edge of the lifetime axis of all maps.")
        f.add_optional_float("lifetime_max", "Lifetime to", 20.0, default=20.0, maximum=1000,
                             step=1, decimals=2, suffix=" ns", auto_text="per map",
                             tooltip="Upper edge of the lifetime axis, shared by all maps "
                                     "so they can be compared directly. Lifetimes outside "
                                     "are dropped.\nUnchecked: every map spans its own "
                                     "data (maps are then not comparable).")
        f.add_optional_float("intensity_max", "Counts to", None, default=1000.0,
                             maximum=1e9, step=10, decimals=1, auto_text="trace maximum",
                             tooltip="Upper edge of the counts axis of all maps (starts "
                                     "at 0).\nUnchecked: maximum of the time trace + 1.")
        f.add_int("lifetime_bins", "Lifetime bins", 100, minimum=1, maximum=1024,
                  tooltip="Number of histogram bins along the lifetime axis.")
        f.add_int("intensity_bins", "Count bins", 100, minimum=1, maximum=1024,
                  tooltip="Number of histogram bins along the counts axis.")
        f.add_float("sigma", "Smoothing", 1.5, maximum=20, step=0.5, decimals=1,
                    suffix=" bins", tooltip="Gaussian smoothing of the 2-D histogram "
                                            "(0 disables it).")
        f.add_choice("mode", "Fit start", FIT_MODES, "fraction",
                     tooltip="How the start of the fit window is chosen (one window for "
                             "all bins of an ROI).")
        f.add_float("fraction", "Edge fraction", 0.95, maximum=1.0, step=0.05, decimals=2,
                    tooltip="Fit starts where the rising edge reaches this fraction of "
                            "the maximum.")
        f.add_float("smoothing", "Edge smoothing", 2.0, maximum=50, step=0.5, decimals=1,
                    tooltip="Smoothing applied before locating the fit start.")
        f.add_float("start", "Start", 0.0, minimum=-1000, maximum=1000, step=0.5,
                    decimals=2, suffix=" ns",
                    tooltip="Fit window start, relative to the histogram maximum.")
        f.add_optional_float("end", "End", None, maximum=1000, step=1, decimals=2,
                             suffix=" ns", auto_text="end of histogram",
                             tooltip="Fit window end, relative to the histogram maximum.")
        layout.addWidget(f)
        f.changed.connect(self._update_fit_fields)
        self._update_fit_fields()
        self.flid_button = self.add_button(
            layout, "Compute FLIDs", self.run_flids, primary=True,
            tooltip="Fit every time bin of every ROI and build the FLID maps. "
                    "This can take a while; use Cancel in the status bar to stop.")

    def _build_export_group(self) -> None:
        layout = self.add_group("CSV export")
        form = QFormLayout()
        self.export_roi = QComboBox()
        self.export_roi.setToolTip("Which ROI's FLID 'Export data (CSV)' writes.")
        self.export_content = QComboBox()
        for data, text in model.EXPORT_CONTENTS:
            self.export_content.addItem(text, data)
        self.export_content.setToolTip("FLID map: the 2-D histogram.\nPer-bin fits: "
                                       "time, counts and fitted lifetimes of every bin.")
        form.addRow("ROI", self.export_roi)
        form.addRow("Content", self.export_content)
        layout.addLayout(form)
        hint = QLabel("Then use 'Export data (CSV)…' above the FLID maps.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(placeholder-text);")
        layout.addWidget(hint)

    def _update_fit_fields(self) -> None:
        mode = self.flid_form.value("mode")
        self.flid_form.set_row_visible("fraction", mode == "fraction")
        self.flid_form.set_row_visible("smoothing", mode in ("fraction", "inflection"))
        self.flid_form.set_row_visible("start", mode == "manual")
        self.flid_form.set_row_visible("end", mode == "manual")

    def _set_enabled(self, enabled: bool) -> None:
        for button in (self.trace_button, self.flid_button, self.add_roi_button):
            button.setEnabled(enabled)

    # ------------------------------------------------------------ parameters
    def trace_params(self) -> TraceParams:
        return TraceParams(**self.trace_form.values())

    def flid_params(self) -> model.FlidParams:
        return model.FlidParams(**self.flid_form.values())

    # ----------------------------------------------------------------- ROIs
    def set_rois(self, rois) -> None:
        """Replace all ROIs (list of ``{x_min, x_max, y_min, y_max}`` dicts)."""
        self.rois = [dict(r) for r in rois]
        self._install_selector()
        self._refresh_roi_list()

    def add_roi(self, roi: dict[str, float]) -> None:
        self.set_rois([*self.rois, roi])

    def add_roi_from_form(self) -> None:
        v = self.roi_form.values()
        if v["x_max"] <= v["x_min"] or v["y_max"] <= v["y_min"]:
            self.ctx.log.warning("ROI not added: 'To' must be larger than 'From'.")
            return
        self.add_roi(v)

    def remove_selected_roi(self) -> None:
        row = self.roi_list.currentRow()
        if 0 <= row < len(self.rois):
            self.set_rois(self.rois[:row] + self.rois[row + 1:])

    def clear_rois(self) -> None:
        self.set_rois([])

    def _sync_from_selector(self) -> None:
        """Copy the selector's boxes after a mouse/keyboard interaction."""
        if self.selector is None:
            return
        boxes = self.selector.get_boxes()
        if boxes != self.rois:
            self.rois = boxes
            self._refresh_roi_list()
        self._decorate_boxes()

    def _refresh_roi_list(self) -> None:
        self.roi_list.clear()
        for i, roi in enumerate(self.rois):
            item = QListWidgetItem(
                f"ROI {i + 1}: {roi['x_min']:.1f}–{roi['x_max']:.1f} s, "
                f"{roi['y_min']:.0f}–{roi['y_max']:.0f} counts")
            item.setForeground(QColor(_mpl_hex(_colour(i))))
            self.roi_list.addItem(item)

    def _install_selector(self) -> None:
        """(Re)attach the box selector to the trace axes and restore the ROIs."""
        if self.selector is not None:
            self.selector.disconnect()
            self.selector.set_boxes([])
            self.selector = None
        axes = self.trace_plot.figure.axes
        if not axes or self.trace is None:
            self._decorate_boxes()
            return
        self.selector = InteractiveBoxSelector(axes[0])
        # Matplotlib calls handlers in connection order, so connecting after the
        # selector lets these read its boxes once it has processed the event.
        canvas = self.trace_plot.canvas
        for cid in self._event_cids:
            canvas.mpl_disconnect(cid)
        self._event_cids = [
            canvas.mpl_connect("button_release_event", lambda _e: self._sync_from_selector()),
            canvas.mpl_connect("key_press_event", lambda _e: self._sync_from_selector()),
        ]
        self.selector.set_boxes(self.rois)
        self._decorate_boxes()

    def _decorate_boxes(self) -> None:
        """Colour the boxes like their FLID maps and number them."""
        for text in self._roi_labels:
            text.remove()
        self._roi_labels = []
        if self.selector is None:
            self.trace_plot.canvas.draw_idle()
            return
        ax = self.selector.ax
        for i, (box, roi) in enumerate(zip(self.selector.boxes, self.rois)):
            if box.get("patch") is not None:
                box["patch"].set_facecolor(_colour(i))
            self._roi_labels.append(ax.text(
                roi["x_min"], roi["y_max"], f" {i + 1}", color=_colour(i), fontweight="bold",
                va="top", ha="left", zorder=6, clip_on=True))
        self.trace_plot.canvas.draw_idle()

    # -------------------------------------------------------------- actions
    def on_file_changed(self, kind: str, handle) -> None:
        if kind != "ptu":
            return
        self.trace = self.result = None
        self.rois = []
        self._refresh_roi_list()
        self._refresh_export_rois()
        if self.selector is not None:
            self.selector.disconnect()
            self.selector = None
        self._roi_labels = []
        self.flid_plot.show_message("Draw ROIs on the time trace and press 'Compute FLIDs'.")
        if handle is None:
            self.file_label.setText("No PTU file loaded.")
            self.trace_plot.show_message("Open a .ptu file to start.")
            self._set_enabled(False)
            return
        self.file_label.setText(handle.name)
        if handle.reader.is_cw:
            self._set_enabled(False)
            self.trace_plot.show_message("This is a continuous-wave file: it has no "
                                         "lifetime axis, so FLIDs are not possible.")
            return
        self._set_enabled(True)
        self.update_trace()

    def update_trace(self) -> None:
        handle = self.ctx.state.ptu
        if handle is None:
            return
        params = self.trace_params()
        self.run_job(f"Time trace of {handle.name}",
                     lambda progress: compute_trace(handle, params), self._trace_done,
                     disable=[self.trace_button, self.flid_button])

    def _trace_done(self, result) -> None:
        old = self.trace
        self.trace = result
        if old is not None and self.rois and old.params.bintime != result.params.bintime:
            factor = result.params.bintime / old.params.bintime
            self.rois = model.scale_rois(self.rois, factor)
            self._refresh_roi_list()
            self.ctx.log.info(f"Bin time changed: ROI count ranges scaled by {factor:g}.")
        self._draw_trace()

    def run_flids(self) -> None:
        handle = self.ctx.state.ptu
        if handle is None or self.trace is None:
            return
        try:
            params = self.flid_params()
            params.lifetime_range()
        except ValueError as exc:
            self.ctx.log.warning(str(exc))
            return
        # Use the bin time the ROIs were drawn with, even if the field changed since.
        trace_params = self.trace.params
        if self.trace_form.value("bintime") != trace_params.bintime:
            self.ctx.log.warning("The bin time field differs from the drawn trace; using "
                                 f"the trace's {trace_params.bintime:g} s. Press 'Update "
                                 "time trace' to change it.")
        rois = list(self.rois)
        self.run_job(f"FLIDs of {handle.name}",
                     lambda progress: model.compute_flids(handle, trace_params, params,
                                                          rois, progress),
                     self._flids_done, disable=[self.flid_button, self.trace_button],
                     cancellable=True)

    def _flids_done(self, result: model.FlidResult) -> None:
        self.result = result
        if result.fallback:
            roi = result.entries[0].roi if result.entries else model.default_roi(result.trace)
            self.ctx.log.info(f"No ROI drawn: using {roi['x_min']:.1f}–{roi['x_max']:.1f} s "
                              "(between the first and last laser trigger).")
        for index, _roi, message in result.skipped:
            self.ctx.log.warning(f"ROI {index + 1} skipped: {message}")
        for entry in result.entries:
            self.ctx.log.info(f"ROI {entry.index + 1}: {entry.n_fitted} of {entry.n_bins} "
                              "time bins fitted.")
        self._refresh_export_rois()
        self._draw_flids()
        self.plots.setCurrentWidget(self.flid_plot)

    def _refresh_export_rois(self) -> None:
        self.export_roi.clear()
        if self.result is None:
            return
        for k, entry in enumerate(self.result.entries):
            name = "Fallback window" if self.result.fallback else f"ROI {entry.index + 1}"
            self.export_roi.addItem(name, k)

    # --------------------------------------------------------------- export
    def _trace_payload(self):
        if self.trace is None:
            return None
        from ..models.ptu import trace_payload

        payload = trace_payload(self.trace)
        payload.metadata["rois"] = [dict(r) for r in self.rois]
        return payload

    def _flid_payload(self):
        if self.result is None or not self.result.entries:
            return None
        k = self.export_roi.currentData()
        entry = self.result.entries[k if k is not None else 0]
        if self.export_content.currentData() == "bins":
            return model.bin_fits_payload(self.result, entry)
        return model.flid_payload(self.result, entry)

    # -------------------------------------------------------------- drawing
    def _draw_trace(self) -> None:
        if self.trace is None:
            return
        if self.selector is not None:
            self.selector.disconnect()
            self.selector = None
        self._roi_labels = []
        panel = self.trace_plot
        panel.clear()
        ax = panel.figure.add_subplot()
        s = self.trace.signal
        ax.plot(s.x, s.y, color="k", linewidth=0.8)
        if self.trace.marker_times is not None:
            for i, t in enumerate(self.trace.marker_times):
                ax.axvline(t, color="0.5", linestyle="--", linewidth=0.8,
                           label="Laser trigger" if i == 0 else None)
            ax.legend(loc="upper right")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Counts / {self.trace.params.bintime * 1e3:g} ms")
        ax.set_title("Left-drag: draw ROI | drag inside/handles: move/resize | "
                     "Delete: remove selected", fontsize="medium")
        ax.grid(True, alpha=0.3)
        panel.draw()
        self._install_selector()

    def _draw_flids(self) -> None:
        result = self.result
        panel = self.flid_plot
        if result is None or not result.entries:
            panel.show_message("None of the ROIs contained photons.")
            return
        panel.clear()
        fig_tt, fig_maps = panel.figure.subfigures(2, 1, height_ratios=[1, 1.6])
        ax_tt = fig_tt.subplots()
        s = result.trace.signal
        ax_tt.plot(s.x, s.y, color="k", linewidth=0.6)
        ax_tt.set_xlabel("Time (s)")
        ax_tt.set_ylabel(f"Counts / {result.trace.params.bintime * 1e3:g} ms")
        ax_tt.set_title(s.label)
        ax_tt.grid(True, alpha=0.3)

        n = len(result.entries)
        ncols = min(n, 4)
        nrows = int(np.ceil(n / ncols))
        axes = np.atleast_1d(fig_maps.subplots(nrows, ncols, sharex=True, sharey=True,
                                               squeeze=False)).ravel()
        for ax, entry in zip(axes, result.entries):
            colour = _colour(entry.index)
            roi = entry.roi
            ax_tt.add_patch(Rectangle((roi["x_min"], roi["y_min"]),
                                      roi["x_max"] - roi["x_min"], roi["y_max"] - roi["y_min"],
                                      edgecolor=colour, facecolor=colour, alpha=0.25,
                                      linewidth=2))
            img = entry.image
            extent = (*img.meta["lifetime_range"], *img.meta["intensity_range"])
            im = ax.imshow(img.data, extent=extent, origin="lower", aspect="auto",
                           cmap="viridis")
            fig_maps.colorbar(im, ax=ax, label=img.units["z"])
            name = "Fallback window" if result.fallback else f"ROI {entry.index + 1}"
            ax.set_title(f"{name}: {roi['x_min']:.0f}–{roi['x_max']:.0f} s "
                         f"({entry.n_fitted} bins)", color=colour)
            ax.set_xlabel(f"Lifetime ({img.units['x']})")
        for ax in axes[len(result.entries):]:
            ax.set_visible(False)
        for row in range(nrows):
            axes[row * ncols].set_ylabel(f"Counts / {result.trace.params.bintime * 1e3:g} ms")
        panel.draw()


def _mpl_hex(colour: str) -> str:
    from matplotlib.colors import to_hex

    return to_hex(colour)
