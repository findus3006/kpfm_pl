"""PTU tab: time trace, on/off states, lifetime fit and lifetime trace."""

from __future__ import annotations

import numpy as np
from matplotlib.widgets import SpanSelector
from PySide6.QtWidgets import QLabel, QTableWidget, QTableWidgetItem, QTabWidget

from ..canvas import PlotPanel
from ..forms import ParamForm
from ..models import ptu as model
from .base import BaseTab

__all__ = ["PTUTab"]


class PTUTab(BaseTab):
    """Explore one ``.ptu`` file: trace, states, lifetime fit, lifetime trace."""

    title = "PL lifetime (PTU)"

    def __init__(self, ctx, parent=None) -> None:
        super().__init__(ctx, parent)
        self.trace: model.TraceResult | None = None
        self.preview_hist = None          # (x_ns, y, label) shown before a fit
        self.fit: model.FitResult | None = None
        self.lifetime: tuple | None = None  # (signal, params...) of the last lifetime trace
        self._span = None

        self.add_hint(
            "1. Open a .ptu file. The time trace is computed automatically.\n"
            "2. Optionally set on/off thresholds (dashed lines on the trace).\n"
            "3. Fit the lifetime of the whole trace or of one state.\n"
            "4. Optionally compute the lifetime for every time bin (slow).\n"
            "Hover over a field for an explanation."
        )
        self._build_file_group()
        self._build_trace_group()
        self._build_state_group()
        self._build_fit_group()
        self._build_lifetime_group()

        self.plots = QTabWidget()
        self.trace_plot = PlotPanel(ctx)
        self.hist_plot = PlotPanel(ctx)
        self.lt_plot = PlotPanel(ctx)
        self.plots.addTab(self.trace_plot, "Time trace")
        self.plots.addTab(self.hist_plot, "Lifetime histogram")
        self.plots.addTab(self.lt_plot, "Lifetime trace")
        self.content_layout.addWidget(self.plots)
        self.trace_plot.set_export_provider(
            lambda: model.trace_payload(self.trace) if self.trace else None)
        self.hist_plot.set_export_provider(
            lambda: model.fit_payload(self.fit) if self.fit else None)
        self.lt_plot.set_export_provider(self._lifetime_payload)
        for panel in (self.trace_plot, self.hist_plot, self.lt_plot):
            panel.show_message("Open a .ptu file to start.")
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
        f.add_float("bintime", "Bin time", 0.05, minimum=1e-4, maximum=600, step=0.01,
                    decimals=4, suffix=" s",
                    tooltip="Width of one time bin. Photons are counted per bin.")
        f.add_float("tstart", "Start", 0.0, maximum=1e6, step=1, decimals=2, suffix=" s",
                    tooltip="Only use photons after this time.")
        f.add_optional_float("tend", "End", None, maximum=1e6, step=1, decimals=2,
                             suffix=" s", auto_text="end of file",
                             tooltip="Only use photons before this time. "
                                     "Unchecked: up to the end of the file.")
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
        layout.addWidget(f)
        self.trace_button = self.add_button(layout, "Update time trace", self.update_trace,
                                            primary=True)

    def _build_state_group(self) -> None:
        layout = self.add_group("On/off states")
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
        layout = self.add_group("Lifetime fit")
        f = self.fit_form = ParamForm()
        f.add_choice("source", "Histogram of", model.HISTOGRAM_SOURCES, "all",
                     tooltip="Which time bins contribute to the fitted lifetime histogram.")
        f.add_int("n_exp", "Exponentials", 2, minimum=1, maximum=4,
                  tooltip="Number of exponential decay components.")
        f.add_choice("mode", "Fit start", model.FIT_MODES, "fraction",
                     tooltip="How the start of the fit window is chosen.\n"
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
        self.fit_button = self.add_button(layout, "Fit lifetime", self.run_fit, primary=True)
        self.fit_table = QTableWidget(0, 2)
        self.fit_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.fit_table.horizontalHeader().setStretchLastSection(True)
        self.fit_table.verticalHeader().setVisible(False)
        self.fit_table.setMinimumHeight(160)
        layout.addWidget(self.fit_table)

    def _build_lifetime_group(self) -> None:
        layout = self.add_group("Lifetime trace")
        f = self.lt_form = ParamForm()
        f.add_float("bintime", "Bin time", 0.5, minimum=1e-3, maximum=600, step=0.1,
                    decimals=3, suffix=" s",
                    tooltip="Each bin of this width is fitted separately. Smaller bins "
                            "give more points but noisier fits and take longer.")
        f.add_int("n_exp", "Exponentials", 1, minimum=1, maximum=4,
                  tooltip="Components per bin fit. The plotted lifetime is their "
                          "intensity-weighted mean.")
        f.add_int("bins", "Lifetime binning", 1, minimum=1, maximum=64,
                  tooltip="Merge this many lifetime channels before fitting "
                          "(helps with few photons per bin).")
        layout.addWidget(f)
        layout.addWidget(self._small("Uses the fit window settings of 'Lifetime fit' and "
                                     "the time window of 'Time trace'."))
        self.lt_button = self.add_button(layout, "Compute lifetime trace",
                                         self.run_lifetime_trace)

    @staticmethod
    def _small(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: palette(placeholder-text);")
        return label

    def _update_fit_fields(self) -> None:
        mode = self.fit_form.value("mode")
        self.fit_form.set_row_visible("fraction", mode == "fraction")
        self.fit_form.set_row_visible("smoothing", mode in ("fraction", "inflection"))
        self.fit_form.set_row_visible("start", mode == "manual")
        self.fit_form.set_row_visible("end", mode == "manual")
        self._install_span()

    def _set_enabled(self, enabled: bool) -> None:
        for button in (self.trace_button, self.fit_button, self.lt_button):
            button.setEnabled(enabled)

    # ------------------------------------------------------------ parameters
    def trace_params(self) -> model.TraceParams:
        return model.TraceParams(**self.trace_form.values())

    def state_params(self) -> model.StateParams:
        v = self.state_form.values()
        return model.StateParams(off=v["off"] / 100.0, on=v["on"] / 100.0)

    def fit_params(self) -> model.FitParams:
        return model.FitParams(**self.fit_form.values())

    def lt_params(self) -> model.LifetimeTraceParams:
        return model.LifetimeTraceParams(**self.lt_form.values())

    # -------------------------------------------------------------- actions
    def on_file_changed(self, kind: str, handle) -> None:
        if kind != "ptu":
            return
        self.trace = self.fit = self.lifetime = self.preview_hist = None
        self.fit_table.setRowCount(0)
        if handle is None:
            self.file_label.setText("No PTU file loaded.")
            self._set_enabled(False)
            return
        self.file_label.setText(handle.name)
        if handle.reader.is_cw:
            self._set_enabled(False)
            for panel in (self.trace_plot, self.hist_plot, self.lt_plot):
                panel.show_message("This is a continuous-wave file: it has no lifetime "
                                   "axis, so time-binned and lifetime analysis are not "
                                   "possible.")
            return
        self._set_enabled(True)
        self.lt_plot.show_message("Press 'Compute lifetime trace' to fit every time bin.")
        self.update_trace()

    def _handle(self):
        return self.ctx.state.ptu

    def update_trace(self) -> None:
        handle = self._handle()
        if handle is None:
            return
        params, states, fit = self.trace_params(), self.state_params(), self.fit_params()

        def job(progress):
            result = model.compute_trace(handle, params)
            hist = model.compute_histogram(handle, params, fit, states)
            return result, hist

        self.run_job(f"Time trace of {handle.name}", job, self._trace_done,
                     disable=[self.trace_button])

    def _trace_done(self, value) -> None:
        self.trace, hist = value
        self._draw_trace()
        # A new trace (other bin time or window) invalidates the previous fit.
        self.fit = None
        self.fit_table.setRowCount(0)
        y = np.asarray(hist.y, dtype=float)
        x_ns = (hist.x - hist.x[int(np.argmax(y))]) * 1e9 if y.size else hist.x
        self.preview_hist = (x_ns, y, hist.label)
        self._draw_histogram()

    def run_fit(self) -> None:
        handle = self._handle()
        if handle is None:
            return
        params, states, fit = self.trace_params(), self.state_params(), self.fit_params()
        self.run_job(f"Lifetime fit of {handle.name}",
                     lambda progress: model.compute_fit(handle, params, fit, states),
                     self._fit_done, disable=[self.fit_button])

    def _fit_done(self, result: model.FitResult) -> None:
        self.fit = result
        y = np.asarray(result.histogram.y, dtype=float)
        self.preview_hist = (result.fit["shifted_histogram_x"], y, result.histogram.label)
        summary = model.fit_summary(result.fit)
        self.fit_table.setRowCount(0)
        for key, value in summary.items():
            row = self.fit_table.rowCount()
            self.fit_table.insertRow(row)
            text = f"{value:.4g}" if isinstance(value, float) else str(value)
            self.fit_table.setItem(row, 0, QTableWidgetItem(key))
            self.fit_table.setItem(row, 1, QTableWidgetItem(text))
        self.fit_table.resizeColumnsToContents()
        self._draw_histogram()
        self.plots.setCurrentWidget(self.hist_plot)
        if not summary["success"]:
            self.ctx.log.warning("The lifetime fit did not converge; check the fit window.")

    def run_lifetime_trace(self) -> None:
        handle = self._handle()
        if handle is None:
            return
        params, fit, lt = self.trace_params(), self.fit_params(), self.lt_params()

        def job(progress):
            signal = model.compute_lifetime_trace(handle, params, fit, lt, progress)
            return signal, str(handle.path), params, fit, lt

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

    # -------------------------------------------------------------- drawing
    def _draw_trace(self) -> None:
        if self.trace is None:
            return
        panel = self.trace_plot
        panel.clear()
        ax = panel.figure.add_subplot()
        s = self.trace.signal
        bintime = self.trace.params.bintime
        ax.plot(s.x, s.y, color="k", linewidth=0.8, label="Time trace")
        if self.trace.marker_times is not None:
            for i, t in enumerate(self.trace.marker_times):
                ax.axvline(t, color="tab:red", linestyle="--", linewidth=0.8,
                           label="Laser trigger" if i == 0 else None)
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
        ax.set_title(s.label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        panel.draw()

    def _draw_histogram(self) -> None:
        if self.preview_hist is None:
            return
        panel = self.hist_plot
        panel.clear()
        ax = panel.figure.add_subplot()
        x, y, label = self.preview_hist
        ax.semilogy(x, np.where(y > 0, y, np.nan), ".", color="0.3", markersize=3,
                    label="Histogram")
        if self.fit is not None:
            fit = self.fit.fit
            curve = np.asarray(fit["curve"], dtype=float)
            if self.fit.params.normalize:
                # Undo the normalisation so the curve overlays the raw counts.
                curve = curve * np.max(y[np.isin(x, fit["t_fit"])])
            ax.semilogy(fit["t_fit"], curve, "-", color="tab:red", linewidth=1.5,
                        label="Fit")
            ax.axvspan(fit["t_fit"][0], fit["t_fit"][-1], color="tab:red", alpha=0.08,
                       label="Fit window")
            taus = ", ".join(f"τ{i} = {t:.3g} ns"
                             for i, t in enumerate(np.atleast_1d(fit["tau"]), 1))
            ax.set_title(f"{label}: {taus}")
        else:
            ax.set_title(label)
        ax.set_xlabel("Time after histogram maximum (ns)")
        ax.set_ylabel("Counts")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        panel.draw()
        self._install_span()

    def _install_span(self) -> None:
        """In manual fit mode, dragging on the histogram sets start/end."""
        if not hasattr(self, "hist_plot"):
            return
        self._span = None
        axes = self.hist_plot.figure.axes
        if not axes or self.fit_form.value("mode") != "manual":
            return

        def on_select(lo: float, hi: float) -> None:
            if hi - lo <= 0:
                return
            self.fit_form.set_value("start", round(lo, 2))
            self.fit_form.set_value("end", round(hi, 2))

        self._span = SpanSelector(axes[0], on_select, "horizontal", useblit=True,
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
        ax.plot(signal.x, signal.y, ".", color="tab:purple", markersize=3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean lifetime (ns)")
        n_ok = int(np.count_nonzero(np.isfinite(signal.y)))
        ax.set_title(f"{signal.label} ({n_ok} of {len(signal)} bins fitted)")
        ax.grid(True, alpha=0.3)
        panel.draw()
