"""Antibunching tab: second-order correlation g2 of two detectors of a T2 file."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QLabel, QTableWidget, QTableWidgetItem, QTabWidget

from ..canvas import PlotPanel
from ..forms import ParamForm
from ..models import g2 as model
from .base import BaseTab
from .rois import RoiEditor

__all__ = ["G2Tab"]

_PLOT_KINDS = (("counts", "Coincidences"), ("normalised", "Normalised g2"))


class G2Tab(BaseTab):
    """Hanbury Brown–Twiss correlation of two detectors, with g2(0)."""

    title = "Antibunching (g2)"

    def __init__(self, ctx, parent=None) -> None:
        super().__init__(ctx, parent)
        self.trace = None                     # (Signal, G2TraceParams) of the detector trace
        self.result: model.G2Result | None = None

        self.add_hint(
            "Second-order correlation g2(τ) of two detectors behind a beam splitter "
            "(Hanbury Brown–Twiss), e.g. to prove single-photon emission. Needs a T2 "
            "file. The file is the one of the 'PL lifetime' tab.\n"
            "1. Choose the two detectors and the bin time of the trace.\n"
            "2. Choose the photons: the whole time window, the on state, or an ROI drawn "
            "on the trace (left-drag draws a box).\n"
            "3. Compute g2. With a laser period (from the sync or entered) g2(0) is the "
            "centre-peak area over the mean side-peak area.")

        self.plots = QTabWidget()
        self.trace_plot = PlotPanel(ctx, navigation="minimal")
        self.g2_plot = PlotPanel(ctx)

        self._build_file_group()
        self._build_trace_group()
        self._build_roi_group()
        self._build_g2_group()

        self.plots.addTab(self.trace_plot, "Detector trace && ROIs")
        self.plots.addTab(self.g2_plot, "g2")
        self.content_layout.addWidget(self.plots)
        self.trace_plot.set_export_provider(self._trace_payload)
        self.g2_plot.set_export_provider(self._g2_payload)
        self.on_file_changed("ptu", ctx.state.ptu)

    # ------------------------------------------------------------- controls
    def _build_file_group(self) -> None:
        layout = self.add_group("File")
        self.file_label = QLabel("No PTU file loaded.")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: palette(placeholder-text);")
        layout.addWidget(self.info_label)
        self.add_button(layout, "Open PTU file…", lambda: self.ctx.open_and_load("ptu"))

    def _build_trace_group(self) -> None:
        layout = self.add_group("Detectors and trace", collapsible=True)
        f = self.trace_form = ParamForm()
        f.add_choice("channel_a", "Detector A", model.detector_options(None), 1,
                     tooltip="Start detector. Positive lags: the photon of B came later.")
        f.add_choice("channel_b", "Detector B", model.detector_options(None), 2,
                     tooltip="Stop detector.")
        f.add_float("bintime", "Bin time", 0.01, minimum=1e-4, maximum=600, step=0.01,
                    decimals=4, suffix=" s",
                    tooltip="Width of the time bins of the trace (photons of both "
                            "detectors) on which the on state and the ROIs are chosen. "
                            "Changing it rescales the count range of existing ROIs. "
                            "Keep it much longer than the lags.")
        layout.addWidget(f)
        self.trace_button = self.add_button(layout, "Update trace", self.update_trace,
                                            primary=True)

    def _build_roi_group(self) -> None:
        layout = self.add_group("Regions of interest (ROIs)", collapsible=True)
        hint = QLabel("Draw boxes on the trace or type them in below. A box selects the "
                      "time bins inside its time window and count range.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(placeholder-text);")
        layout.addWidget(hint)
        self.roi_editor = RoiEditor(self.ctx, self.trace_plot, layout, self)
        self.roi_editor.changed.connect(self._rois_changed)

    def _build_g2_group(self) -> None:
        layout = self.add_group("Correlation", collapsible=True)
        f = self.g2_form = ParamForm()
        f.add_choice("region", "Photons of", model.region_options([]), "all",
                     tooltip="Whole time window, the on state (bins above the threshold, "
                             "e.g. the bright state of a blinking emitter) or one ROI.")
        f.add_float("on_percent", "  On state ≥", 60.0, minimum=0.1, maximum=100, step=5,
                    decimals=1, suffix=" % of max",
                    tooltip="Time bins with at least this share of the largest count in "
                            "the time window form the on state.")
        f.add_float("tstart", "  From", 0.0, maximum=1e6, step=1, decimals=3, suffix=" s",
                    tooltip="Start of the time window (whole window and on state).")
        f.add_optional_float("tend", "  To", None, default=10.0, maximum=1e6, step=1,
                             decimals=3, suffix=" s", auto_text="end of file",
                             tooltip="End of the time window (whole window and on state).")
        f.add_heading("Lags")
        f.add_optional_float("max_lag_ns", "Max lag", None, default=500.0, minimum=1e-3,
                             maximum=1e7, step=10, decimals=3, suffix=" ns",
                             tooltip="Largest lag of the correlogram. Automatic: 5.5 laser "
                                     "periods (centre peak and five side peaks on each "
                                     "side), or 500 ns without a period.")
        f.add_optional_float("lag_bin_ns", "Lag bin", None, default=0.5, minimum=1e-3,
                             maximum=1e5, step=0.1, decimals=3, suffix=" ns",
                             tooltip="Width of one lag bin (rounded to whole time tags). "
                                     "Automatic: 1/200 of the laser period, or 1 ns.")
        f.add_optional_float("period_ns", "Laser period", None, default=100.0,
                             minimum=1e-3, maximum=1e7, step=1, decimals=4, suffix=" ns",
                             auto_text="from the sync",
                             tooltip="Spacing of the peaks for the pulsed g2(0). Automatic: "
                                     "measured from the laser sync of the file. Files "
                                     "without a sync (PicoHarp 300 T2) need it here; "
                                     "without it g2(0) of continuous excitation is "
                                     "computed.")
        f.add_optional_float("delay_ns", "Delay", None, default=0.0, minimum=-1e6,
                             maximum=1e6, step=0.5, decimals=3, suffix=" ns",
                             tooltip="Lag of the centre peak (detector and cable delays). "
                                     "Automatic: found from the peak comb (pulsed) or 0.")
        f.add_optional_float("window_ns", "Peak window ±", None, default=10.0,
                             minimum=1e-3, maximum=1e6, step=1, decimals=3, suffix=" ns",
                             tooltip="Half-width of the area around each peak. Automatic: "
                                     "half a period, so every coincidence counts (pulsed), "
                                     "or the lag bin at the delay (continuous).")
        f.add_choice("plot", "Plot", _PLOT_KINDS, "counts",
                     tooltip="Coincidences per lag bin, or normalised to uncorrelated "
                             "light (N_A N_B Δτ / T).")
        f.changed.connect(self._update_g2_fields)
        layout.addWidget(f)
        self._update_g2_fields()
        self.g2_button = self.add_button(layout, "Compute g2", self.run_g2, primary=True)
        self.result_table = QTableWidget(0, 2)
        self.result_table.setHorizontalHeaderLabels(["Result", "Value"])
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.result_table.setMinimumHeight(220)
        layout.addWidget(self.result_table)

    def _update_g2_fields(self) -> None:
        region = self.g2_form.value("region") or "all"
        window = not str(region).startswith("roi:")
        self.g2_form.set_row_visible("on_percent", region == "on")
        for key in ("tstart", "tend"):
            self.g2_form.set_row_visible(key, window)
        if self.result is not None and self.sender() is self.g2_form:
            self._draw_g2()
        self._draw_trace()

    # ------------------------------------------------------------ parameters
    def trace_params(self) -> model.G2TraceParams:
        v = self.trace_form.values()
        return model.G2TraceParams(bintime=v["bintime"], channel_a=int(v["channel_a"]),
                                   channel_b=int(v["channel_b"]))

    def g2_params(self) -> model.G2Params:
        v = self.g2_form.values()
        region, roi = str(v.pop("region") or "all"), 0
        v.pop("plot")
        if region.startswith("roi:"):
            region, roi = "roi", int(region[4:]) - 1
        return model.G2Params(region=region, roi=roi, **v)

    @property
    def rois(self) -> list[dict[str, float]]:
        return self.roi_editor.rois

    def set_rois(self, rois) -> None:
        self.roi_editor.set_rois(rois)

    def _rois_changed(self) -> None:
        self.g2_form.set_choices("region", model.region_options(self.rois))

    # -------------------------------------------------------------- actions
    def _handle(self):
        return self.ctx.state.ptu

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (self.trace_button, self.g2_button, self.roi_editor.add_button):
            widget.setEnabled(enabled)

    def on_file_changed(self, kind: str, handle) -> None:
        if kind != "ptu":
            return
        self.trace = self.result = None
        self.result_table.setRowCount(0)
        self.roi_editor.detach()
        self.roi_editor.set_rois([])
        options = model.detector_options(handle)
        self.trace_form.set_choices("channel_a", options)
        self.trace_form.set_choices("channel_b", options)
        if len(options) > 1:
            self.trace_form.set_value("channel_a", options[0][0])
            self.trace_form.set_value("channel_b", options[1][0])
        self.file_label.setText("No PTU file loaded." if handle is None else handle.name)
        self.info_label.setText("" if handle is None or handle.reader.is_cw
                                else model.describe_g2_file(handle))
        reason = model.unavailable_reason(handle)
        if reason is not None:
            self._set_enabled(False)
            for panel in (self.trace_plot, self.g2_plot):
                panel.show_message(reason)
            return
        self._set_enabled(True)
        self.g2_plot.show_message("Press 'Compute g2' to correlate the two detectors.")
        self.update_trace()

    def update_trace(self) -> None:
        handle = self._handle()
        if handle is None or model.unavailable_reason(handle) is not None:
            return
        params = self.trace_params()
        if params.channel_a == params.channel_b:
            self.ctx.log.warning("Choose two different detectors.")
            return
        old = self.trace[1] if self.trace is not None else None

        def done(signal) -> None:
            if old is not None and old.bintime != params.bintime and self.rois:
                factor = params.bintime / old.bintime
                self.roi_editor.set_rois([{**r, "y_min": r["y_min"] * factor,
                                           "y_max": r["y_max"] * factor} for r in self.rois])
                self.ctx.log.info(f"Bin time changed: ROI count ranges scaled by {factor:g}.")
            self.trace = (signal, params)
            self._draw_trace()
            self.plots.setCurrentWidget(self.trace_plot)

        self.run_job(f"Detector trace of {handle.name}",
                     lambda progress: model.compute_detector_trace(handle, params), done,
                     disable=[self.trace_button])

    def run_g2(self) -> None:
        handle = self._handle()
        if handle is None or model.unavailable_reason(handle) is not None:
            return
        try:
            trace_params, params = self.trace_params(), self.g2_params()
            params.validate()
        except ValueError as exc:
            self.ctx.log.warning(str(exc))
            return
        rois = list(self.rois)
        self.run_job(f"g2 of {handle.name}",
                     lambda progress: model.compute_g2(handle, trace_params, params, rois,
                                                       progress),
                     self._g2_done, disable=[self.g2_button, self.trace_button],
                     cancellable=True)

    def _g2_done(self, result: model.G2Result) -> None:
        self.result = result
        meta = result.signal.meta
        self.ctx.log.info(f"g2 ({result.region_name}): "
                          f"{model.g2_title(result).splitlines()[-1]}")
        if meta["method"] == "continuous" and meta["period"] is None:
            self.ctx.log.info("No laser period known: g2(0) of continuous excitation. "
                              "Enter the laser period for the pulsed g2(0).")
        rows = model.result_rows(result)
        table = self.result_table
        table.setRowCount(len(rows))
        for r, (label, value, tip) in enumerate(rows):
            for c, text in enumerate((label, value)):
                item = QTableWidgetItem(text)
                item.setToolTip(tip)
                table.setItem(r, c, item)
        table.resizeColumnsToContents()
        self._draw_g2()
        self.plots.setCurrentWidget(self.g2_plot)

    # -------------------------------------------------------------- drawing
    def _draw_trace(self) -> None:
        if self.trace is None:
            return
        signal, params = self.trace
        self.roi_editor.detach()
        panel = self.trace_plot
        panel.clear()
        ax = panel.figure.add_subplot()
        ax.plot(signal.x, signal.y, color="k", linewidth=0.8,
                label=f"Detectors {params.channel_a} + {params.channel_b}")
        try:
            g2 = self.g2_params()
            bounds, name = model.selection(g2, signal, self.rois)
        except ValueError:
            g2, bounds = None, None
        if g2 is not None and g2.region != "roi":
            if bounds["y_min"] is not None:
                ax.axhline(bounds["y_min"], color="tab:green", linestyle=":",
                           label=f"on ≥ {g2.on_percent:g} %")
            if g2.tstart > 0:
                ax.axvline(g2.tstart, color="tab:blue", linestyle="--", linewidth=0.8,
                           label="Time window")
            if g2.tend is not None:
                ax.axvline(g2.tend, color="tab:blue", linestyle="--", linewidth=0.8,
                           label=None if g2.tstart > 0 else "Time window")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Counts / {params.bintime * 1e3:g} ms")
        ax.set_title(f"{signal.label}\nLeft-drag: draw ROI | drag inside/handles: "
                     "move/resize | Delete: remove selected", fontsize="medium")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        panel.draw()
        self.roi_editor.attach(ax)

    def _draw_g2(self) -> None:
        result = self.result
        if result is None:
            return
        s = result.signal
        meta = s.meta
        normalised = self.g2_form.value("plot") == "normalised"
        lag_ns = np.asarray(s.x, dtype=float) * 1e9
        y = np.asarray(meta["g2"] if normalised else s.y, dtype=float)
        panel = self.g2_plot
        panel.clear()
        ax = panel.figure.add_subplot()
        ax.step(lag_ns, y, where="mid", color="k", linewidth=0.8, label="g2")
        delay, window = meta["delay"] * 1e9, meta["window"] * 1e9
        ax.axvspan(delay - window, delay + window, color="tab:red", alpha=0.15,
                   label="Centre peak")
        if meta["method"] == "pulsed":
            period = meta["period"] * 1e9
            for k, key in enumerate(sorted(meta["side_areas"])):
                centre = delay + key * period
                ax.axvspan(centre - window, centre + window, color="tab:blue", alpha=0.08,
                           label="Side peaks" if k == 0 else None)
        if normalised:
            ax.axhline(1.0, color="0.5", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Lag τ (ns)")
        ax.set_ylabel("g2(τ), normalised" if normalised
                      else f"Coincidences / {meta['lag_bin'] * 1e9:.3g} ns")
        ax.set_title(model.g2_title(result), fontsize="medium")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize="small")
        panel.draw()

    # --------------------------------------------------------------- export
    def _trace_payload(self):
        handle = self._handle()
        if self.trace is None or handle is None:
            return None
        return model.trace_payload(handle, *self.trace)

    def _g2_payload(self):
        return None if self.result is None else model.g2_payload(self.result)
