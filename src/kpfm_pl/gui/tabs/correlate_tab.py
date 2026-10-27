"""Correlate tab: align a KPFM signal (Zurich .txt) with a PL time trace (.ptu)."""

from __future__ import annotations

import dataclasses
import html

from matplotlib.transforms import blended_transform_factory
from matplotlib.widgets import SpanSelector
from PySide6.QtWidgets import QLabel, QTableWidget, QTableWidgetItem, QTabWidget

from ..canvas import PlotPanel
from ..forms import ParamForm
from ..models import correlate as model
from ..models import ptu as ptu_model
from .base import BaseTab

__all__ = ["CorrelateTab"]

_PRESETS = (("current", "Current (PTU markers, trigger pulses)"),
            ("legacy", "Legacy (no PTU markers, trigger level)"))

_SUMMARY_LABELS = {
    "time_shift_s": ("Time shift", "s"),
    "grid_step_s": ("Grid step", "s"),
    "kpfm_triggers": ("Zurich triggers", ""),
    "timetrace_triggers": ("PTU triggers", ""),
    "aligned_on_trigger": ("Aligned on trigger no.", ""),
    "window_start_s": ("Window start", "s"),
    "window_end_s": ("Window end", "s"),
    "window_duration_s": ("Window duration", "s"),
    "triggers_in_window": ("Triggers in window", ""),
}


class CorrelateTab(BaseTab):
    """Find the laser triggers in both files, align the signals, cut a window."""

    title = "Correlate"

    def __init__(self, ctx, parent=None) -> None:
        super().__init__(ctx, parent)
        self.triggers: model.TriggerResult | None = None
        self.triggers_stale = True
        self.result: model.MatchResult | None = None
        self._generation = 0          # ignores results of superseded jobs
        self._updating_window = False
        self._span = None

        self.add_hint(
            "1. Open a .ptu (PL) and a Zurich .txt (KPFM) file of the same measurement.\n"
            "2. Press 'Find triggers' and check that both files show the same laser "
            "triggers (numbered lines).\n"
            "3. Press 'Align' to shift the time trace onto the KPFM signal and cut out "
            "the window. Drag on the aligned plot to choose another window.\n"
            "Hover over a field for an explanation."
        )
        self._build_file_group()
        self._build_ptu_group()
        self._build_zurich_group()
        self._build_match_group()
        self._build_window_group()

        self.plots = QTabWidget()
        self.trigger_plot = PlotPanel(ctx)
        self.aligned_plot = PlotPanel(ctx)
        self.plots.addTab(self.trigger_plot, "Triggers")
        self.plots.addTab(self.aligned_plot, "Aligned")
        self.content_layout.addWidget(self.plots)
        self.trigger_plot.set_export_provider(
            lambda: model.trigger_payload(self.triggers) if self.triggers else None)
        self.aligned_plot.set_export_provider(
            lambda: model.aligned_payload(self.result) if self.result else None)
        self._reset_plots()
        self._update_enabled()

    # ------------------------------------------------------------- controls
    def _build_file_group(self) -> None:
        layout = self.add_group("Files")
        self.ptu_label = QLabel("PTU: not loaded")
        self.zurich_label = QLabel("Zurich: not loaded")
        for label in (self.ptu_label, self.zurich_label):
            label.setWordWrap(True)
            layout.addWidget(label)
        self.add_button(layout, "Open PTU file…", lambda: self.ctx.open_and_load("ptu"))
        self.add_button(layout, "Open Zurich file…", lambda: self.ctx.open_and_load("zurich"))
        f = self.preset_form = ParamForm()
        f.add_choice("preset", "Recording type", _PRESETS, "current",
                     tooltip="Current recordings store the laser triggers as PTU markers "
                             "and as short pulses in the Zurich trigger stream.\n"
                             "Legacy recordings have no PTU markers and a trigger level "
                             "that drops when the laser switches on.\nChoosing a type "
                             "sets 'Laser markers' and 'Flank' accordingly.")
        f.changed.connect(self._apply_preset)
        layout.addWidget(f)

    def _build_ptu_group(self) -> None:
        layout = self.add_group("PL time trace (PTU)")
        f = self.ptu_form = ParamForm()
        f.add_float("bintime", "Bin time", 0.05, minimum=1e-4, maximum=600, step=0.01,
                    decimals=4, suffix=" s",
                    tooltip="Width of one time bin of the PL time trace.")
        f.add_bool("laser_pulse", "Laser markers", True, text="recorded in marker channel",
                   tooltip="Checked (current recordings): laser triggers are read from "
                           "the PTU marker channel.\nUnchecked (legacy files without "
                           "markers): the edges of the non-zero part of the trace are "
                           "used as triggers.")
        f.add_int("marker_channel", "Marker channel", 4, minimum=1, maximum=15,
                  tooltip="PicoHarp marker channel carrying the laser trigger.")
        f.changed.connect(self._mark_stale)
        layout.addWidget(f)

    def _build_zurich_group(self) -> None:
        layout = self.add_group("KPFM signal (Zurich)")
        f = self.zurich_form = ParamForm()
        f.add_choice("kpfm_stream", "KPFM stream", [],
                     tooltip="Stream holding the KPFM signal (usually the first one).")
        f.add_choice("trigger_stream", "Trigger stream", [],
                     tooltip="Stream holding the laser trigger (usually the second one).")
        f.add_float("bintime", "Averaging", 0.05, minimum=1e-4, maximum=600, step=0.01,
                    decimals=4, suffix=" s",
                    tooltip="The KPFM signal is averaged over bins of this width. "
                            "Usually the same as the PTU bin time.")
        f.add_choice("method", "Trigger detection", model.TRIGGER_METHODS, "slope",
                     tooltip="Slope: steepest steps of the trigger stream (recommended).\n"
                             "Threshold: where the stream crosses 10 % of its range.")
        f.add_choice("flank", "Flank", model.FLANKS, "rising",
                     tooltip="Which steps count as laser triggers. Rising matches the PTU "
                             "markers of current recordings; legacy files need 'Both'.")
        f.add_float("slope_threshold", "Slope threshold", 0.1, minimum=0.001, maximum=1.0,
                    step=0.05, decimals=3,
                    tooltip="A step counts as a trigger when its slope exceeds this "
                            "fraction of the steepest step.")
        f.changed.connect(self._mark_stale)
        f.changed.connect(self._update_zurich_fields)
        layout.addWidget(f)
        self._update_zurich_fields()
        self.trigger_button = self.add_button(layout, "1. Find triggers", self.find_triggers,
                                              primary=True)
        self.trigger_label = QLabel("")
        self.trigger_label.setWordWrap(True)
        layout.addWidget(self.trigger_label)

    def _build_match_group(self) -> None:
        layout = self.add_group("Alignment")
        f = self.match_form = ParamForm()
        f.add_int("align", "Align on trigger no.", 1, minimum=1, maximum=10_000,
                  tooltip="The time trace is shifted so that this trigger coincides in "
                          "both files (counted from 1, see the numbers in the Triggers "
                          "plot).")
        f.add_optional_float("dt", "Grid step", None, minimum=1e-5, maximum=60, step=0.01,
                             default=0.05, decimals=5, suffix=" s", auto_text="auto",
                             tooltip="Spacing of the shared time grid. Unchecked: the "
                                     "finer of the two bin widths.")
        f.add_int("kpfm_smoothing", "KPFM smoothing", 20, minimum=1, maximum=10_000,
                  tooltip="Moving-average window (in samples) applied to the KPFM signal. "
                          "1 = no smoothing.")
        f.add_int("timetrace_smoothing", "Trace smoothing", 1, minimum=1, maximum=10_000,
                  tooltip="Moving-average window (in samples) applied to the time trace. "
                          "1 = no smoothing.")
        layout.addWidget(f)

    def _build_window_group(self) -> None:
        layout = self.add_group("Window")
        f = self.window_form = ParamForm()
        for key, label, kind in (("start", "Start at", "first"), ("stop", "End at", "last")):
            f.add_choice(key, label, model.BOUND_KINDS, kind,
                         tooltip="Where the extracted window begins/ends: a trigger, a "
                                 "time on the aligned axis, or the edge of the region "
                                 "where both signals have data.")
            f.add_int(f"{key}_trigger", "  Trigger no.", 1, minimum=1, maximum=10_000,
                      tooltip="Trigger number (counted from 1).")
            f.add_float(f"{key}_time", "  Time", 0.0, minimum=-1e6, maximum=1e6, step=1,
                        decimals=3, suffix=" s", tooltip="Time on the aligned (KPFM) axis.")
        f.changed.connect(self._update_window_fields)
        f.changed.connect(self._window_changed)
        layout.addWidget(f)
        d = self.display_form = ParamForm()
        d.add_bool("scale", "Display", True, text="scale time trace to 0–1",
                   tooltip="Min-max scale the time trace in the window for plotting only. "
                           "Exported data are never scaled.")
        d.changed.connect(self._draw_aligned)
        layout.addWidget(d)
        self._update_window_fields()
        self.align_button = self.add_button(layout, "2. Align and extract", self.run_align,
                                            primary=True)
        self.window_label = QLabel("")
        self.window_label.setWordWrap(True)
        layout.addWidget(self.window_label)
        self.result_table = QTableWidget(0, 2)
        self.result_table.setHorizontalHeaderLabels(["Result", "Value"])
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setMinimumHeight(200)
        layout.addWidget(self.result_table)

    def _update_zurich_fields(self) -> None:
        self.zurich_form.set_row_visible("slope_threshold",
                                         self.zurich_form.value("method") == "slope")

    def _update_window_fields(self) -> None:
        for key in ("start", "stop"):
            kind = self.window_form.value(key)
            self.window_form.set_row_visible(f"{key}_trigger", kind == "trigger")
            self.window_form.set_row_visible(f"{key}_time", kind == "time")

    def _apply_preset(self) -> None:
        legacy = self.preset_form.value("preset") == "legacy"
        self.ptu_form.set_value("laser_pulse", not legacy)
        self.zurich_form.set_value("flank", "both" if legacy else "rising")

    def _update_enabled(self) -> None:
        ready = self.ctx.state.ptu is not None and self.ctx.state.zurich is not None
        self.trigger_button.setEnabled(ready)
        self.align_button.setEnabled(ready)

    def _reset_plots(self) -> None:
        self.trigger_plot.show_message("Open a .ptu and a Zurich .txt file, then press "
                                       "'Find triggers'.")
        self.aligned_plot.show_message("Press 'Align and extract' to align both signals.")

    # ------------------------------------------------------------ parameters
    def trace_params(self) -> ptu_model.TraceParams:
        return ptu_model.TraceParams(**self.ptu_form.values())

    def zurich_params(self) -> model.ZurichParams:
        return model.ZurichParams(**self.zurich_form.values())

    def match_params(self) -> model.MatchParams:
        v = self.match_form.values()
        v["align"] -= 1
        return model.MatchParams(**v)

    def window_params(self) -> model.WindowParams:
        return model.WindowParams(**self.window_form.values())

    # -------------------------------------------------------------- actions
    def on_file_changed(self, kind: str, handle) -> None:
        if kind not in ("ptu", "zurich"):
            return
        state = self.ctx.state
        self.ptu_label.setText(f"PTU: {state.ptu.name}" if state.ptu else "PTU: not loaded")
        self.zurich_label.setText(f"Zurich: {state.zurich.name}" if state.zurich
                                  else "Zurich: not loaded")
        if kind == "zurich" and handle is not None:
            self._fill_streams(handle.reader.stream_names())
        self._generation += 1
        self.triggers, self.result = None, None
        self.triggers_stale = True
        self.trigger_label.setText("")
        self.window_label.setText("")
        self.result_table.setRowCount(0)
        self._reset_plots()
        self._update_enabled()
        if state.ptu is not None and state.zurich is not None:
            self.find_triggers()

    def _fill_streams(self, names: list[str]) -> None:
        options = [(name, name) for name in names]
        form = self.zurich_form
        form.blockSignals(True)
        for key, default in (("kpfm_stream", 0), ("trigger_stream", 1)):
            previous = form.value(key)
            form.set_choices(key, options)
            if previous not in names and names:
                form.set_value(key, names[min(default, len(names) - 1)])
        form.blockSignals(False)

    def _mark_stale(self) -> None:
        self.triggers_stale = True

    def _handles(self):
        state = self.ctx.state
        return state.ptu, state.zurich

    def find_triggers(self) -> None:
        ptu, zurich = self._handles()
        if ptu is None or zurich is None:
            return
        trace_p, zurich_p = self.trace_params(), self.zurich_params()
        self._generation += 1
        generation = self._generation
        self.run_job("Finding laser triggers",
                     lambda progress: model.compute_triggers(ptu, zurich, trace_p, zurich_p),
                     lambda result: self._triggers_done(result, generation),
                     disable=[self.trigger_button, self.align_button])

    def _triggers_done(self, result: model.TriggerResult, generation: int) -> None:
        if generation != self._generation:
            return
        self.triggers = result
        self.triggers_stale = False
        self.result = None
        self._show_trigger_status(result)
        self._draw_triggers()
        self.aligned_plot.show_message("Press 'Align and extract' to align both signals.")
        self.plots.setCurrentWidget(self.trigger_plot)

    def _show_trigger_status(self, result: model.TriggerResult) -> None:
        n_k = len(result.kpfm_triggers or [])
        n_t = len(result.timetrace_triggers or [])
        problems = model.trigger_problems(result)
        text = f"Zurich: {n_k} trigger(s), PTU: {n_t} trigger(s)."
        if problems:
            for problem in problems:
                self.ctx.log.warning(problem)
            text += "".join(f"<br><span style='color:#b36b00'>⚠ {html.escape(p)}</span>"
                            for p in problems)
        else:
            text += "<br><span style='color:#2e7d32'>✓ Ready to align.</span>"
        self.trigger_label.setText(text)

    def run_align(self) -> None:
        ptu, zurich = self._handles()
        if ptu is None or zurich is None:
            return
        trace_p, zurich_p = self.trace_params(), self.zurich_params()
        match_p, window_p = self.match_params(), self.window_params()
        existing = None if self.triggers_stale else self.triggers
        self._generation += 1
        generation = self._generation

        def job(progress):
            triggers = existing or model.compute_triggers(ptu, zurich, trace_p, zurich_p)
            return triggers, model.compute_match(triggers, match_p, window_p)

        self.run_job("Aligning KPFM and PL", job,
                     lambda value: self._align_done(value, generation),
                     disable=[self.trigger_button, self.align_button])

    def _align_done(self, value, generation: int) -> None:
        if generation != self._generation:
            return
        triggers, result = value
        if triggers is not self.triggers:
            self.triggers = triggers
            self.triggers_stale = False
            self._show_trigger_status(triggers)
            self._draw_triggers()
        self.result = result
        self.window_label.setText("")
        self._show_summary()
        self._draw_aligned()
        self.plots.setCurrentWidget(self.aligned_plot)

    def _window_changed(self) -> None:
        """Re-cut the existing alignment when the window settings change."""
        if self._updating_window or self.result is None:
            return
        window_p = self.window_params()
        try:
            extracted, start, stop = model.extract_window(self.result.match, window_p)
        except ValueError as exc:
            self.window_label.setText(f"<span style='color:#c00000'>{html.escape(str(exc))}"
                                      "</span>")
            return
        self.window_label.setText("")
        self.result = dataclasses.replace(self.result, window=extracted, start=start,
                                          stop=stop, window_params=window_p)
        self._show_summary()
        self._draw_aligned()

    def _show_summary(self) -> None:
        self.result_table.setRowCount(0)
        for key, value in model.match_summary(self.result).items():
            label, unit = _SUMMARY_LABELS.get(key, (key, ""))
            text = f"{value:.4g}" if isinstance(value, float) else str(value)
            row = self.result_table.rowCount()
            self.result_table.insertRow(row)
            self.result_table.setItem(row, 0, QTableWidgetItem(label))
            self.result_table.setItem(row, 1, QTableWidgetItem(f"{text} {unit}".strip()))
        self.result_table.resizeColumnsToContents()

    # -------------------------------------------------------------- drawing
    @staticmethod
    def _trigger_lines(ax, points, color: str, label: str) -> None:
        """Dashed vertical lines with the 1-based trigger number on top."""
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for i, point in enumerate(points or []):
            ax.axvline(point["time"], color=color, linestyle="--", linewidth=0.9,
                       label=label if i == 0 else None)
            ax.text(point["time"], 1.0, f" {i + 1}", transform=trans, color=color,
                    va="top", ha="left", fontsize=9, fontweight="bold")

    def _draw_triggers(self) -> None:
        if self.triggers is None:
            return
        r = self.triggers
        panel = self.trigger_plot
        panel.clear()
        ax_k, ax_t = panel.figure.subplots(2, 1)
        ax_s = ax_k.twinx()
        ax_s.plot(r.trigger_signal.x, r.trigger_signal.y, color="0.7", linewidth=0.8)
        ax_s.set_ylabel(f"Trigger stream ({r.trigger_signal.units['y']})", color="0.5")
        ax_k.set_zorder(ax_s.get_zorder() + 1)
        ax_k.patch.set_visible(False)
        ax_k.plot(r.kpfm.x, r.kpfm.y, color="tab:red", linewidth=0.8, label="KPFM")
        self._trigger_lines(ax_k, r.kpfm_triggers, "k", "Laser trigger")
        ax_k.set_ylabel(f"KPFM ({r.kpfm.units['y']})", color="tab:red")
        ax_k.set_title(f"{r.kpfm.label}: {len(r.kpfm_triggers or [])} trigger(s)")
        ax_t.plot(r.timetrace.x, r.timetrace.y, color="tab:blue", linewidth=0.8)
        self._trigger_lines(ax_t, r.timetrace_triggers, "k", "Laser trigger")
        ax_t.set_ylabel(f"Counts / {r.trace_params.bintime * 1e3:g} ms", color="tab:blue")
        ax_t.set_title(f"{r.timetrace.label}: {len(r.timetrace_triggers or [])} trigger(s)")
        for ax in (ax_k, ax_t):
            ax.set_xlabel("Time in own file (s)")
            ax.grid(True, alpha=0.3)
        panel.draw()

    def _draw_aligned(self) -> None:
        if self.result is None:
            return
        r = self.result
        m = r.match
        full_k, full_t = m.kpfm_signal(), m.timetrace_signal()
        win_k, win_t = r.window["kpfm"], r.window["timetrace"]
        scale = self.display_form.value("scale")
        panel = self.aligned_plot
        panel.clear()
        ax = panel.figure.add_subplot()
        ax2 = ax.twinx()
        # KPFM (left axis) is drawn above the time trace; its background is
        # made transparent so the twin axis stays visible underneath.
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax2.axvspan(r.start, r.stop, color="0.5", alpha=0.08, zorder=0)
        ax.plot(full_k.x, full_k.y, color="tab:red", alpha=0.25, linewidth=0.8)
        ax.plot(win_k.x, win_k.y, color="tab:red", linewidth=1.0, label="KPFM")
        if scale:
            t_full = model.scaled(full_t.y, win_t.y)
            t_win = model.scaled(win_t.y, win_t.y)
            ax2.set_ylabel("Time trace (scaled to window)", color="tab:blue")
        else:
            t_full, t_win = full_t.y, win_t.y
            ax2.set_ylabel(f"Time trace ({win_t.units['y']})", color="tab:blue")
        ax2.plot(full_t.x, t_full, color="tab:blue", alpha=0.2, linewidth=0.8)
        ax2.plot(win_t.x, t_win, color="tab:blue", alpha=0.8, linewidth=0.8,
                 label="PL time trace")
        self._trigger_lines(ax, m.triggers, "k", "Laser trigger")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"KPFM ({win_k.units['y']})", color="tab:red")
        ax.set_title(f"Aligned on trigger {r.match_params.align + 1}: time trace shifted "
                     f"by {m.offset:.3f} s; window {r.start:.2f}–{r.stop:.2f} s")
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        panel.figure.legend(handles + handles2, labels + labels2, loc="outside lower center",
                            ncol=3, fontsize=8)
        panel.draw()
        self._install_span(ax)

    def _install_span(self, ax) -> None:
        """Dragging on the aligned plot sets the window to that time range."""

        def on_select(lo: float, hi: float) -> None:
            if hi - lo <= 0:
                return
            self._updating_window = True
            try:
                self.window_form.set_values({"start": "time", "start_time": round(lo, 3),
                                             "stop": "time", "stop_time": round(hi, 3)})
            finally:
                self._updating_window = False
            self._window_changed()

        self._span = SpanSelector(ax, on_select, "horizontal", useblit=True,
                                  props={"alpha": 0.15, "facecolor": "tab:orange"},
                                  interactive=False, button=1)
