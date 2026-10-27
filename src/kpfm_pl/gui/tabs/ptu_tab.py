"""PL lifetime tab: time trace with ROIs, lifetime fits, lifetime trace and FLIDs.

All analyses work on the time trace shown in the *Time trace & ROIs* plot.
ROIs drawn there can be fitted (one, all combined or each separately),
restrict the lifetime trace and define the FLID maps. Because an ROI's count
range only has a meaning at the bin time it was drawn at, everything that
uses ROIs runs with the trace's bin time.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
from matplotlib.widgets import SpanSelector
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
)

from ..canvas import PlotPanel
from ..fit_options import FitOptionsDialog, describe_options
from ..forms import ParamForm
from ..models import flid as flid_model
from ..models import ptu as model
from .base import BaseTab
from .rois import RoiEditor, roi_colour

__all__ = ["PTUTab"]

_START_KINDS = (("time", "Time"), ("trigger", "Trigger"))
_END_KINDS = (("end", "End of file"), ("time", "Time"), ("trigger", "Trigger"))

#: Row labels of the fit-results table (keys of ``models.ptu.fit_summary``).
_SUMMARY_LABELS = {
    "offset": "Offset",
    "reduced_chi_squared": "χ²ᵣ (Poisson)",
    "r_squared": "R²",
    "success": "Converged",
    "fit_start_ns": "Fit start (ns)",
    "fit_end_ns": "Fit end (ns)",
    "fit_points": "Fit points",
}
_CHI_TOOLTIP = ("Reduced chi-squared with Poisson weights (counts - fit)² / counts.\n"
                "≈ 1: the fit describes the data within photon noise.\n"
                "Clearly > 1: systematic deviations (try more exponentials or another "
                "fit window; see the residuals below the histogram).\n"
                "< 1: suspiciously good, e.g. overfitting or few photons.")


def _summary_label(key: str) -> str:
    if key.startswith("tau") and key.endswith("_ns"):
        return f"τ{key[3:-3]} (ns)"
    return _SUMMARY_LABELS.get(key, key)


def _chi_colour(chi: float) -> QColor | None:
    if not np.isfinite(chi):
        return None
    return QColor("#2e7d32") if chi <= 1.5 else QColor("#b36b00")


class PTUTab(BaseTab):
    """Explore one ``.ptu`` file: trace, ROIs, lifetime fits, lifetime trace, FLIDs."""

    title = "PL lifetime (PTU)"

    def __init__(self, ctx, parent=None) -> None:
        super().__init__(ctx, parent)
        self.trace: model.TraceResult | None = None
        self.preview_hist = None          # (x_ns, y, label) shown before a fit
        self.fits: list[model.FitResult] = []
        self.lifetime: tuple | None = None  # (signal, payload arguments...) of the last trace
        self.flid_result: flid_model.FlidResult | None = None
        self.fit_options = model.FitOptions()
        self._span = None
        self._hist_ax = None
        # Chosen window triggers survive a new file, whose list is still empty.
        self._chosen_triggers: dict[str, int | None] = {"start_trigger": None,
                                                        "end_trigger": None}

        self.add_hint(
            "1. Open a .ptu file. The time trace is computed automatically; choose its "
            "time window by time or by laser trigger.\n"
            "2. Optionally draw regions (ROIs) on the trace: left-drag draws a box, "
            "drag inside moves it, drag a handle resizes it, Delete removes the "
            "selected box.\n"
            "3. Fit the lifetime of the whole trace, an on/off state or the ROIs.\n"
            "4. Optionally fit every time bin (lifetime trace) or build FLID maps.\n"
            "Click a group title to fold it. Hover over a field for an explanation."
        )
        self.plots = QTabWidget()
        self.trace_plot = PlotPanel(ctx, navigation="minimal")
        self.hist_plot = PlotPanel(ctx)
        self.lt_plot = PlotPanel(ctx)
        self.flid_plot = PlotPanel(ctx)

        self._build_file_group()
        self._build_trace_group()
        self._build_roi_group()
        self._build_state_group()
        self._build_fit_group()
        self._build_lifetime_group()
        self._build_flid_group()

        self.plots.addTab(self.trace_plot, "Time trace && ROIs")
        self.plots.addTab(self.hist_plot, "Lifetime histogram")
        self.plots.addTab(self.lt_plot, "Lifetime trace")
        self.plots.addTab(self.flid_plot, "FLID maps")
        self.content_layout.addWidget(self.plots)
        self.trace_plot.set_export_provider(self._trace_payload)
        self.hist_plot.set_export_provider(
            lambda: model.fits_payload(self.fits) if self.fits else None)
        self.lt_plot.set_export_provider(self._lifetime_payload)
        self.flid_plot.set_export_provider(self._flid_payload)
        for panel in (self.trace_plot, self.hist_plot, self.lt_plot, self.flid_plot):
            panel.show_message("Open a .ptu file to start.")
        self._set_enabled(False)

    @property
    def fit(self) -> model.FitResult | None:
        """The (first) current lifetime fit."""
        return self.fits[0] if self.fits else None

    @property
    def rois(self) -> list[dict[str, float]]:
        return self.roi_editor.rois

    def set_rois(self, rois) -> None:
        """Replace all ROIs (list of ``{x_min, x_max, y_min, y_max}`` dicts)."""
        self.roi_editor.set_rois(rois)

    # ------------------------------------------------------------- controls
    def _build_file_group(self) -> None:
        layout = self.add_group("File")
        self.file_label = QLabel("No PTU file loaded.")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)
        self.add_button(layout, "Open PTU file…", lambda: self.ctx.open_and_load("ptu"))

    def _build_trace_group(self) -> None:
        layout = self.add_group("Time trace", collapsible=True)
        f = self.trace_form = ParamForm()
        f.add_float("bintime", "Bin time", 0.05, minimum=1e-4, maximum=600, step=0.01,
                    decimals=4, suffix=" s",
                    tooltip="Width of one time bin. Photons are counted per bin. ROIs, "
                            "ROI lifetime traces and FLIDs use this bin time; changing "
                            "it rescales the count range of existing ROIs.")
        f.add_choice("start_kind", "Start at", _START_KINDS, "time",
                     tooltip="Begin the trace at a time or at a laser trigger.")
        f.add_float("tstart", "  Time", 0.0, maximum=1e6, step=1, decimals=3, suffix=" s",
                    tooltip="Only use photons after this time.")
        f.add_choice("start_trigger", "  Trigger", [],
                     tooltip="Begin at this laser trigger (numbers as in the plot). "
                             "The list is filled once a trace was computed.")
        f.add_choice("end_kind", "End at", _END_KINDS, "end",
                     tooltip="End the trace at the end of the file, a time or a laser "
                             "trigger.")
        f.add_float("tend", "  Time", 60.0, maximum=1e6, step=1, decimals=3, suffix=" s",
                    tooltip="Only use photons before this time.")
        f.add_choice("end_trigger", "  Trigger", [],
                     tooltip="End at this laser trigger (numbers as in the plot).")
        f.add_float("dark_counts", "Dark counts", 0.0, maximum=1e9, step=0.25, decimals=3,
                    suffix=" /s", tooltip="Detector dark counts per second. Subtracted "
                                          "as a constant offset in the lifetime fits.")
        f.add_bool("laser_pulse", "Laser markers", True,
                   text="recorded in marker channel",
                   tooltip="Checked (default, current recordings): laser on/off triggers "
                           "are read from the PTU marker channel.\nUnchecked: legacy files "
                           "without markers. The non-zero part of the trace is centred and "
                           "its two edges are used as triggers.")
        f.add_int("marker_channel", "Marker channel", 4, minimum=1, maximum=15,
                  tooltip="PicoHarp marker channel carrying the laser trigger.")
        f.changed.connect(self._update_trace_fields)
        layout.addWidget(f)
        self._update_trace_fields()
        self.trace_button = self.add_button(layout, "Update time trace", self.update_trace,
                                            primary=True)

    def _build_roi_group(self) -> None:
        layout = self.add_group("Regions of interest (ROIs)", collapsible=True)
        layout.addWidget(self._small("Draw boxes on the time trace, or type them in "
                                     "below. Used by the lifetime fit, the lifetime "
                                     "trace and the FLIDs."))
        self.roi_editor = RoiEditor(self.ctx, self.trace_plot, layout, self)
        self.roi_editor.changed.connect(self._rois_changed)
        self.roi_list = self.roi_editor.list

    def _build_state_group(self) -> None:
        layout = self.add_group("On/off states", collapsible=True, collapsed=True)
        f = self.state_form = ParamForm()
        f.add_float("on", "On state above", 100.0, maximum=100.0, step=5, decimals=1,
                    suffix=" %", tooltip="Time bins with at least this fraction of the "
                                         "maximum count form the 'on' state.\n100 % together "
                                         "with off = 0 % uses every bin as 'on'.")
        f.add_float("off", "Off state below", 0.0, maximum=100.0, step=5, decimals=1,
                    suffix=" %", tooltip="Time bins below this fraction of the maximum "
                                         "count form the 'off' state.")
        layout.addWidget(f)
        f.changed.connect(self._draw_trace)

    def _build_fit_group(self) -> None:
        layout = self.add_group("Lifetime fit", collapsible=True)
        f = self.fit_form = ParamForm()
        f.add_choice("source", "Histogram of", self._source_options(), "all",
                     tooltip="Which time bins contribute to the fitted lifetime histogram: "
                             "the whole trace, an on/off state, one ROI, all ROIs "
                             "combined, or one fit per ROI.")
        f.add_int("n_exp", "Exponentials", 2, minimum=1, maximum=model.MAX_COMPONENTS,
                  tooltip="Number of exponential decay components.")
        f.add_choice("mode", "Fit start", model.FIT_MODES, "fraction",
                     tooltip="How the start of the fit window is chosen. Also used by "
                             "the lifetime trace and the FLIDs.\n"
                             "Manual: drag on the histogram plot or enter start/end.")
        f.add_float("fraction", "Edge fraction", 0.95, maximum=1.0, step=0.05, decimals=2,
                    tooltip="Fit starts where the rising edge reaches this fraction "
                            "of the maximum.")
        f.add_float("smoothing", "Smoothing", 2.0, maximum=50, step=0.5, decimals=1,
                    tooltip="Smoothing applied to the histogram before locating the start.")
        f.add_float("start", "Start", 0.0, minimum=-1000, maximum=1000, step=0.5,
                    decimals=2, suffix=" ns", tooltip="Fit window start, relative to the "
                                                      "histogram maximum.")
        f.add_optional_float("end", "End", None, maximum=1000, step=1, decimals=2,
                             suffix=" ns", auto_text="end of histogram",
                             tooltip="Fit window end, relative to the histogram maximum.")
        f.add_bool("normalize", "Normalise", False, text="scale histogram to 1",
                   tooltip="Divide the histogram by its maximum before fitting.")
        layout.addWidget(f)
        f.changed.connect(self._update_fit_fields)
        self._update_fit_fields()
        self.options_button = self.add_button(
            layout, "Advanced fit settings…", self.edit_fit_options,
            tooltip="Initial guesses and bounds of lifetimes and amplitudes, Poisson "
                    "weighting and robust losses. Shared by the lifetime fit, the "
                    "lifetime trace and the FLIDs.")
        self.options_label = self._small("")
        layout.addWidget(self.options_label)
        self._show_options()
        self.fit_button = self.add_button(layout, "Fit lifetime", self.run_fit, primary=True)
        self.fit_table = QTableWidget(0, 2)
        self.fit_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.fit_table.horizontalHeader().setStretchLastSection(True)
        self.fit_table.verticalHeader().setVisible(False)
        self.fit_table.setMinimumHeight(200)
        layout.addWidget(self.fit_table)

    def _build_lifetime_group(self) -> None:
        layout = self.add_group("Lifetime trace", collapsible=True, collapsed=True)
        f = self.lt_form = ParamForm()
        f.add_choice("region", "Region", self._region_options(), "all",
                     tooltip="Region of the lifetime trace only (FLID has its own): the "
                             "whole time window of the trace, or only the time bins "
                             "inside one ROI (then with the trace's bin time).")
        f.add_float("bintime", "Bin time", 0.5, minimum=1e-3, maximum=600, step=0.1,
                    decimals=3, suffix=" s",
                    tooltip="Each bin of this width is fitted separately. Smaller bins "
                            "give more points but noisier fits and take longer.")
        f.add_int("n_exp", "Exponentials", 1, minimum=1, maximum=model.MAX_COMPONENTS,
                  tooltip="Components per bin fit. The plotted lifetime is their "
                          "intensity-weighted mean.")
        f.add_int("bins", "Lifetime binning", 1, minimum=1, maximum=64,
                  tooltip="Merge this many lifetime channels before fitting "
                          "(helps with few photons per bin).")
        f.changed.connect(self._update_lt_fields)
        layout.addWidget(f)
        self.lt_note = self._small("")
        layout.addWidget(self.lt_note)
        self._update_lt_fields()
        self.lt_button = self.add_button(layout, "Compute lifetime trace",
                                         self.run_lifetime_trace)

    def _build_flid_group(self) -> None:
        layout = self.add_group("FLID", collapsible=True, collapsed=True)
        self.flid_note = self._small("")
        layout.addWidget(self.flid_note)
        f = self.flid_form = ParamForm()
        f.add_choice("region", "Region", self._flid_region_options(), "each",
                     tooltip="What the FLID maps show (independent of the lifetime "
                             "trace's region):\nEach ROI: one map per ROI; without ROIs "
                             "the window between the first and last laser trigger.\n"
                             "Whole time window: one map of the displayed trace, all "
                             "count levels.\nROI n: only that ROI.")
        f.add_int("n_exp", "Exponentials", 1, minimum=1, maximum=model.MAX_COMPONENTS,
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
        layout.addWidget(f)
        self.flid_button = self.add_button(
            layout, "Compute FLIDs", self.run_flids, primary=True,
            tooltip="Fit every time bin of every ROI and build the FLID maps. "
                    "This can take a while; use Cancel in the status bar to stop.")
        form = QFormLayout()
        self.export_roi = QComboBox()
        self.export_roi.setToolTip("Which ROI's FLID 'Export data (CSV)' writes.")
        self.export_content = QComboBox()
        for data, text in flid_model.EXPORT_CONTENTS:
            self.export_content.addItem(text, data)
        self.export_content.setToolTip("FLID map: the 2-D histogram.\nPer-bin fits: "
                                       "time, counts and fitted lifetimes of every bin.")
        form.addRow("CSV: ROI", self.export_roi)
        form.addRow("CSV: content", self.export_content)
        layout.addLayout(form)
        self._update_flid_note()

    @staticmethod
    def _small(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: palette(placeholder-text);")
        return label

    # ----------------------------------------------------------- field state
    def _update_trace_fields(self) -> None:
        f = self.trace_form
        start, end = f.value("start_kind"), f.value("end_kind")
        f.set_row_visible("tstart", start == "time")
        f.set_row_visible("start_trigger", start == "trigger")
        f.set_row_visible("tend", end == "time")
        f.set_row_visible("end_trigger", end == "trigger")

    def _update_fit_fields(self) -> None:
        mode = self.fit_form.value("mode")
        self.fit_form.set_row_visible("fraction", mode == "fraction")
        self.fit_form.set_row_visible("smoothing", mode in ("fraction", "inflection"))
        self.fit_form.set_row_visible("start", mode == "manual")
        self.fit_form.set_row_visible("end", mode == "manual")
        self._install_span()

    def _update_lt_fields(self) -> None:
        roi = self.lt_form.value("region") != "all"
        self.lt_form.set_row_visible("bintime", not roi)
        bintime = self._trace_bintime()
        self.lt_note.setText(
            "Region applies to the lifetime trace only. "
            + (f"ROI: uses the time-trace bin time ({bintime:g} s). " if roi else "")
            + "Uses the fit window of 'Lifetime fit', the advanced fit settings and the "
              "time window of the displayed trace.")

    def _update_flid_note(self) -> None:
        self.flid_note.setText(
            f"Uses the time-trace bin time ({self._trace_bintime():g} s), the fit window "
            "of 'Lifetime fit' and the advanced fit settings. FLIDs usually need bins "
            "of ≥ 0.1 s. 'Each ROI' without ROIs uses the window between the first and "
            "last laser trigger.")

    def _trace_bintime(self) -> float:
        return self.trace.params.bintime if self.trace else self.trace_form.value("bintime")

    def _source_options(self) -> list[tuple[str, str]]:
        rois = self.roi_editor.rois if hasattr(self, "roi_editor") else []
        options = list(model.HISTOGRAM_SOURCES)
        options += [(f"roi:{i + 1}", f"ROI {i + 1}") for i in range(len(rois))]
        if rois:
            options += list(model.ROI_SOURCES)
        return options

    def _region_options(self) -> list[tuple[str, str]]:
        rois = self.roi_editor.rois if hasattr(self, "roi_editor") else []
        return [("all", "Whole time window")] + [(f"roi:{i + 1}", f"ROI {i + 1}")
                                                 for i in range(len(rois))]

    def _flid_region_options(self) -> list[tuple[str, str]]:
        rois = self.roi_editor.rois if hasattr(self, "roi_editor") else []
        return ([("each", "Each ROI (one map per ROI)"), ("window", "Whole time window")]
                + [(f"roi:{i + 1}", f"ROI {i + 1}") for i in range(len(rois))])

    def _rois_changed(self) -> None:
        self.fit_form.set_choices("source", self._source_options())
        self.lt_form.set_choices("region", self._region_options())
        self.flid_form.set_choices("region", self._flid_region_options())

    def _fill_triggers(self) -> None:
        f = self.trace_form
        for key in self._chosen_triggers:
            if f.value(key) is not None:
                self._chosen_triggers[key] = f.value(key)
        options = model.trigger_options(self.trace.trigger_times if self.trace else None)
        numbers = [n for n, _ in options]
        defaults = {"start_trigger": numbers[0] if numbers else None,
                    "end_trigger": numbers[-1] if numbers else None}
        for key in self._chosen_triggers:
            f.set_choices(key, options)
            chosen = self._chosen_triggers[key]
            f.set_value(key, chosen if chosen in numbers else defaults[key])

    def _set_enabled(self, enabled: bool) -> None:
        for button in (self.trace_button, self.fit_button, self.lt_button,
                       self.flid_button, self.roi_editor.add_button):
            button.setEnabled(enabled)

    def _show_options(self) -> None:
        self.options_label.setText(f"Advanced: {describe_options(self.fit_options)}")

    def edit_fit_options(self) -> None:
        dialog = FitOptionsDialog(self.fit_options, self)
        if dialog.exec():
            self.set_fit_options(dialog.options())

    def set_fit_options(self, options: model.FitOptions) -> None:
        options.validate()
        self.fit_options = options
        self._show_options()
        self.ctx.log.info(f"Advanced fit settings: {describe_options(options)}.")

    # ------------------------------------------------------------ parameters
    def trace_params(self) -> model.TraceParams:
        """Time-trace parameters as entered (raises ``ValueError`` if incomplete)."""
        v = self.trace_form.values()
        for key, chosen in self._chosen_triggers.items():
            if v[key] is None:  # list not filled yet (new file): keep the choice
                v[key] = chosen
        start = v["start_trigger"] if v["start_kind"] == "trigger" else None
        end = v["end_trigger"] if v["end_kind"] == "trigger" else None
        if (v["start_kind"] == "trigger" and start is None) or (
                v["end_kind"] == "trigger" and end is None):
            raise ValueError("No laser triggers are known for this file, so the time "
                             "window cannot start or end at a trigger.")
        return model.TraceParams(
            bintime=v["bintime"], tstart=v["tstart"] if v["start_kind"] == "time" else 0.0,
            tend=v["tend"] if v["end_kind"] == "time" else None,
            laser_pulse=v["laser_pulse"], marker_channel=v["marker_channel"],
            dark_counts=v["dark_counts"], start_trigger=start, end_trigger=end)

    def analysis_params(self) -> model.TraceParams | None:
        """Parameters of the displayed trace (the ROIs belong to it).

        Only the dark counts are taken from the form, since they do not change
        the trace. Other edits need 'Update time trace' first; a warning says so.
        """
        if self.trace is None:
            return None
        dark = self.trace_form.value("dark_counts")
        params = dataclasses.replace(self.trace.params, dark_counts=dark)
        try:
            entered = self.trace_params()
        except ValueError:
            entered = None
        if entered != params:
            self.ctx.log.warning("The time-trace settings differ from the displayed trace; "
                                 "the displayed trace is analysed. Press 'Update time "
                                 "trace' to apply them.")
        return params

    def state_params(self) -> model.StateParams:
        v = self.state_form.values()
        return model.StateParams(off=v["off"] / 100.0, on=v["on"] / 100.0)

    def fit_params(self) -> model.FitParams:
        v = self.fit_form.values()
        source = v.pop("source") or "all"
        roi = 0
        if source.startswith("roi:"):
            source, roi = "roi", int(source[4:]) - 1
        return model.FitParams(source=source, roi=roi, **v)

    def lt_params(self) -> model.LifetimeTraceParams:
        v = self.lt_form.values()
        region = v.pop("region") or "all"
        roi = None if region == "all" else int(region[4:]) - 1
        return model.LifetimeTraceParams(roi=roi, **v)

    def flid_params(self) -> flid_model.FlidParams:
        window = self.fit_params().window_kwargs()
        v = self.flid_form.values()
        region = v.pop("region") or "each"
        roi = 0
        if region.startswith("roi:"):
            region, roi = "roi", int(region[4:]) - 1
        return flid_model.FlidParams(**v, **window, region=region, roi=roi)

    # -------------------------------------------------------------- actions
    def on_file_changed(self, kind: str, handle) -> None:
        if kind != "ptu":
            return
        self.trace = self.lifetime = self.preview_hist = self.flid_result = None
        self.fits = []
        self.fit_table.setRowCount(0)
        self.roi_editor.detach()
        self.roi_editor.set_rois([])
        self._fill_triggers()
        self._refresh_export_rois()
        if handle is None:
            self.file_label.setText("No PTU file loaded.")
            self._set_enabled(False)
            for panel in (self.trace_plot, self.hist_plot, self.lt_plot, self.flid_plot):
                panel.show_message("Open a .ptu file to start.")
            return
        self.file_label.setText(handle.name)
        if handle.reader.is_cw:
            self._set_enabled(False)
            for panel in (self.trace_plot, self.hist_plot, self.lt_plot, self.flid_plot):
                panel.show_message("This is a continuous-wave file: it has no lifetime "
                                   "axis, so time-binned and lifetime analysis are not "
                                   "possible.")
            return
        self._set_enabled(True)
        self.lt_plot.show_message("Press 'Compute lifetime trace' to fit every time bin.")
        self.flid_plot.show_message("Draw ROIs on the time trace and press 'Compute FLIDs'.")
        self.update_trace()

    def _handle(self):
        return self.ctx.state.ptu

    def update_trace(self) -> None:
        handle = self._handle()
        if handle is None:
            return
        try:
            params = self.trace_params()
        except ValueError as exc:
            self.ctx.log.warning(str(exc))
            return
        states, fit = self.state_params(), self.fit_params()
        # Counts per bin grow with the bin time, so ROI count ranges are rescaled.
        factor = (params.bintime / self.trace.params.bintime
                  if self.trace is not None and self.rois else 1.0)
        rois = flid_model.scale_rois(self.rois, factor)

        def job(progress):
            result = model.compute_trace(handle, params)
            return result, model.preview_histogram(handle, params, fit, states, rois), factor

        self.run_job(f"Time trace of {handle.name}", job, self._trace_done,
                     disable=[self.trace_button])

    def _trace_done(self, value) -> None:
        self.trace, hist, factor = value
        if factor != 1.0 and self.rois:
            self.roi_editor.set_rois(flid_model.scale_rois(self.rois, factor))
            self.ctx.log.info(f"Bin time changed: ROI count ranges scaled by {factor:g}.")
        self._fill_triggers()
        self._draw_trace()
        # A new trace (other bin time or window) invalidates the previous fit.
        self.fits = []
        self.fit_table.setRowCount(0)
        y = np.asarray(hist.y, dtype=float)
        x_ns = (hist.x - hist.x[int(np.argmax(y))]) * 1e9 if y.size else hist.x
        self.preview_hist = (x_ns, y, hist.label)
        self._draw_histogram()
        self._update_lt_fields()
        self._update_flid_note()

    def run_fit(self) -> None:
        handle = self._handle()
        params = self.analysis_params()
        if handle is None or params is None:
            return
        states, fit = self.state_params(), self.fit_params()
        rois, options = list(self.rois), self.fit_options

        def job(progress):
            if fit.source == "each":
                return model.compute_roi_fits(handle, params, fit, states, rois, options)
            return [model.compute_fit(handle, params, fit, states, rois, options)], []

        self.run_job(f"Lifetime fit of {handle.name}", job, self._fit_done,
                     disable=[self.fit_button])

    def _fit_done(self, value) -> None:
        results, skipped = value
        for index, message in skipped:
            self.ctx.log.warning(f"ROI {index + 1} not fitted: {message}")
        if not results:
            self.ctx.log.warning("No ROI could be fitted.")
            return
        self.fits = results
        first = results[0]
        self.preview_hist = (first.fit["shifted_histogram_x"],
                             np.asarray(first.histogram.y, dtype=float), first.histogram.label)
        self._show_fit_table()
        self._draw_histogram()
        self.plots.setCurrentWidget(self.hist_plot)
        for result in results:
            if not result.fit.get("success", False):
                self.ctx.log.warning(f"The lifetime fit of {result.histogram.label} did not "
                                     "converge; check the fit window and settings.")

    def _show_fit_table(self) -> None:
        table = self.fit_table
        summaries = [model.fit_summary(r.fit) for r in self.fits]
        headers = (["Value"] if len(self.fits) == 1
                   else [f"ROI {r.roi_index + 1}" for r in self.fits])
        table.clear()
        table.setColumnCount(1 + len(headers))
        table.setHorizontalHeaderLabels(["Parameter", *headers])
        keys = list(dict.fromkeys(k for s in summaries for k in s))
        table.setRowCount(len(keys))
        for row, key in enumerate(keys):
            label = QTableWidgetItem(_summary_label(key))
            if key == "reduced_chi_squared":
                label.setToolTip(_CHI_TOOLTIP)
            table.setItem(row, 0, label)
            for col, summary in enumerate(summaries, 1):
                value = summary.get(key)
                text = "" if value is None else (f"{value:.4g}" if isinstance(value, float)
                                                 else str(value))
                item = QTableWidgetItem(text)
                if key == "reduced_chi_squared":
                    item.setToolTip(_CHI_TOOLTIP)
                    colour = _chi_colour(value)
                    if colour is not None:
                        item.setForeground(QBrush(colour))
                table.setItem(row, col, item)
        table.resizeColumnsToContents()

    def run_lifetime_trace(self) -> None:
        handle = self._handle()
        params = self.analysis_params()
        if handle is None or params is None:
            return
        fit, lt = self.fit_params(), self.lt_params()
        rois, options = list(self.rois), self.fit_options

        def job(progress):
            signal = model.compute_lifetime_trace(handle, params, fit, lt, progress,
                                                  rois=rois, options=options)
            return signal, str(handle.path), params, fit, lt, options, rois

        self.run_job(f"Lifetime trace of {handle.name}", job, self._lifetime_done,
                     disable=[self.lt_button], cancellable=True)

    def _lifetime_done(self, value) -> None:
        self.lifetime = value
        self._draw_lifetime()
        self.plots.setCurrentWidget(self.lt_plot)

    def _lifetime_payload(self):
        if self.lifetime is None:
            return None
        return model.lifetime_trace_payload(*self.lifetime)

    def run_flids(self) -> None:
        handle = self._handle()
        params = self.analysis_params()
        if handle is None or params is None:
            return
        try:
            flid = self.flid_params()
            flid.lifetime_range()
        except ValueError as exc:
            self.ctx.log.warning(str(exc))
            return
        rois, options = list(self.rois), self.fit_options
        self.run_job(f"FLIDs of {handle.name}",
                     lambda progress: flid_model.compute_flids(handle, params, flid, rois,
                                                               progress, options),
                     self._flids_done, disable=[self.flid_button, self.trace_button],
                     cancellable=True)

    def _flids_done(self, result: flid_model.FlidResult) -> None:
        self.flid_result = result
        if result.fallback:
            roi = (result.entries[0].roi if result.entries
                   else flid_model.default_roi(result.trace))
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
        if self.flid_result is None:
            return
        for k, entry in enumerate(self.flid_result.entries):
            self.export_roi.addItem(self.flid_result.entry_name(entry), k)

    # --------------------------------------------------------------- export
    def _trace_payload(self):
        if self.trace is None:
            return None
        payload = model.trace_payload(self.trace)
        payload.metadata["rois"] = [dict(r) for r in self.rois]
        return payload

    def _flid_payload(self):
        if self.flid_result is None or not self.flid_result.entries:
            return None
        k = self.export_roi.currentData()
        entry = self.flid_result.entries[k if k is not None else 0]
        if self.export_content.currentData() == "bins":
            return flid_model.bin_fits_payload(self.flid_result, entry)
        return flid_model.flid_payload(self.flid_result, entry)

    # -------------------------------------------------------------- drawing
    def _trigger_numbers(self) -> list[tuple[int, float]]:
        """``(file-wide trigger number, time)`` of the triggers inside the trace."""
        marks = self.trace.marker_times
        if marks is None or not len(marks):
            return []
        times = self.trace.trigger_times
        if times is None or not len(times):
            return list(enumerate(map(float, marks), 1))
        idx = np.clip(np.searchsorted(times, marks), 1, len(times) - 1) if len(times) > 1 \
            else np.zeros(len(marks), dtype=int)
        if len(times) > 1:
            left_closer = np.abs(marks - times[idx - 1]) <= np.abs(times[idx] - marks)
            idx = np.where(left_closer, idx - 1, idx)
        return [(int(i) + 1, float(t)) for i, t in zip(idx, marks)]

    def _draw_trace(self) -> None:
        if self.trace is None:
            return
        self.roi_editor.detach()
        panel = self.trace_plot
        panel.clear()
        ax = panel.figure.add_subplot()
        s = self.trace.signal
        bintime = self.trace.params.bintime
        ax.plot(s.x, s.y, color="k", linewidth=0.8, label="Time trace")
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for k, (number, t) in enumerate(self._trigger_numbers()):
            ax.axvline(t, color="tab:red", linestyle="--", linewidth=0.8,
                       label="Laser trigger" if k == 0 else None)
            ax.text(t, 1.0, f" {number}", transform=trans, color="tab:red", va="top",
                    ha="left", fontsize=8, fontweight="bold", clip_on=True)
        states = self.state_params()
        if len(s.y) and not (states.off == 0 and states.on == 1):
            top = float(np.max(s.y))
            ax.axhline(int(states.on * top), color="tab:green", linestyle=":",
                       label=f"on ≥ {states.on:.0%}")
            if states.off > 0:
                ax.axhline(int(states.off * top), color="tab:blue", linestyle=":",
                           label=f"off < {states.off:.0%}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Counts / {bintime * 1e3:g} ms")
        ax.set_title(f"{s.label}\nLeft-drag: draw ROI | drag inside/handles: move/resize | "
                     "Delete: remove selected", fontsize="medium")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        panel.draw()
        self.roi_editor.attach(ax)

    def _draw_histogram(self) -> None:
        panel = self.hist_plot
        if self.preview_hist is None and not self.fits:
            return
        panel.clear()
        if not self.fits:
            ax = self._hist_ax = panel.figure.add_subplot()
            x, y, label = self.preview_hist
            ax.semilogy(x, np.where(y > 0, y, np.nan), ".", color="0.3", markersize=3,
                        label="Histogram")
            ax.set_title(label)
            ax.set_xlabel("Time after histogram maximum (ns)")
        else:
            ax, ax_res = panel.figure.subplots(2, 1, sharex=True, height_ratios=[3, 1])
            self._hist_ax = ax
            multi = len(self.fits) > 1
            for result in self.fits:
                self._plot_fit(ax, ax_res, result, multi)
            # Keep the log axis on the data; a fit tail must not squash it.
            counts = np.concatenate([np.asarray(r.histogram.y, dtype=float)
                                     for r in self.fits])
            positive = counts[counts > 0]
            if positive.size:
                ax.set_ylim(positive.min() * 0.5, positive.max() * 2.0)
            ax_res.axhline(0.0, color="k", linewidth=0.8)
            ax_res.set_ylabel("Residuals (σ)")
            ax_res.set_xlabel("Time after histogram maximum (ns)")
            ax_res.grid(True, alpha=0.3)
            if multi:
                ax.set_title("Lifetime fits per ROI")
            else:
                fit = self.fit.fit
                taus = ", ".join(f"τ{i} = {t:.3g} ns"
                                 for i, t in enumerate(np.atleast_1d(fit["tau"]), 1))
                ax.set_title(f"{self.fit.histogram.label}: {taus}\n"
                             f"{model.quality_text(fit)}")
        ax.set_ylabel("Counts")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize="small")
        panel.draw()
        self._install_span()

    @staticmethod
    def _plot_fit(ax, ax_res, result: model.FitResult, multi: bool) -> None:
        fit = result.fit
        x = np.asarray(fit["shifted_histogram_x"], dtype=float)
        y = np.asarray(result.histogram.y, dtype=float)
        coloured = multi or result.roi_index is not None
        colour = roi_colour(result.roi_index) if coloured else "tab:red"
        name = f"ROI {result.roi_index + 1}" if result.roi_index is not None else ""
        ax.semilogy(x, np.where(y > 0, y, np.nan), ".", markersize=2 if multi else 3,
                    color=colour if multi else "0.3", alpha=0.5 if multi else 1.0,
                    label=None if multi else "Histogram")
        curve = np.asarray(fit["curve"], dtype=float)
        if result.params.normalize:
            # Undo the normalisation so the curve overlays the raw counts.
            curve = curve * np.max(y[np.isin(x, fit["t_fit"])])
        label = (f"{name}: τ = " + ", ".join(f"{t:.3g}" for t in np.atleast_1d(fit["tau"]))
                 + f" ns, {model.quality_text(fit)}") if multi else "Fit"
        ax.semilogy(fit["t_fit"], curve, "-", color=colour, linewidth=1.5, label=label)
        if not multi:
            ax.axvspan(fit["t_fit"][0], fit["t_fit"][-1], color=colour, alpha=0.08,
                       label="Fit window")
        residuals = fit.get("quality", {}).get("weighted_residuals")
        if residuals is not None:
            ax_res.plot(fit["t_fit"], residuals, "-", color=colour, linewidth=0.7)

    def _install_span(self) -> None:
        """In manual fit mode, dragging on the histogram sets start/end."""
        if not hasattr(self, "hist_plot"):
            return
        self._span = None
        if (self._hist_ax not in self.hist_plot.figure.axes
                or self.fit_form.value("mode") != "manual"):
            return

        def on_select(lo: float, hi: float) -> None:
            if hi - lo <= 0:
                return
            self.fit_form.set_value("start", round(lo, 2))
            self.fit_form.set_value("end", round(hi, 2))

        self._span = SpanSelector(self._hist_ax, on_select, "horizontal", useblit=True,
                                  props={"alpha": 0.15, "facecolor": "tab:orange"},
                                  interactive=False, button=1)

    def _draw_lifetime(self) -> None:
        if self.lifetime is None:
            return
        signal = self.lifetime[0]
        panel = self.lt_plot
        panel.clear()
        ax = panel.figure.add_subplot()
        counts = signal.meta.get("counts")
        if counts is not None and len(counts) == len(signal):
            ax2 = ax.twinx()
            ax2.plot(signal.x, counts, color="0.75", linewidth=0.8, zorder=0)
            ax2.set_ylabel("Counts per bin", color="0.5")
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)
        lt = self.lifetime[4]
        colour = roi_colour(lt.roi) if lt.roi is not None else "tab:purple"
        ax.plot(signal.x, signal.y, ".", color=colour, markersize=3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean lifetime (ns)")
        n_ok = int(np.count_nonzero(np.isfinite(signal.y)))
        ax.set_title(f"{signal.label} ({n_ok} of {len(signal)} bins fitted)")
        ax.grid(True, alpha=0.3)
        panel.draw()

    def _draw_flids(self) -> None:
        result = self.flid_result
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
            name = result.entry_name(entry)
            colour = roi_colour(entry.index) if name.startswith("ROI") else "0.4"
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
            ax.set_title(f"{name}: {roi['x_min']:.0f}–{roi['x_max']:.0f} s "
                         f"({entry.n_fitted} bins)", color=colour)
            ax.set_xlabel(f"Lifetime ({img.units['x']})")
        for ax in axes[len(result.entries):]:
            ax.set_visible(False)
        for row in range(nrows):
            axes[row * ncols].set_ylabel(f"Counts / {result.trace.params.bintime * 1e3:g} ms")
        panel.draw()
