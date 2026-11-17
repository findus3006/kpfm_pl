"""Offscreen tests of the merged PL lifetime tab: trigger window, ROI fits, options."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QSettings  # noqa: E402

from kpfm_pl.gui.app import MainWindow  # noqa: E402
from kpfm_pl.gui.models import ptu as model  # noqa: E402

DATA_DIR = Path(__file__).parents[1] / "examples" / "Test data"
PULSE = DATA_DIR / "ptu_test2_LaserPulse(DR1_800nW_1200_ChargeAccu_000).ptu"

pytestmark = pytest.mark.skipif(not PULSE.exists(), reason="PTU test file missing")

ROI_A = {"x_min": 300.0, "x_max": 340.0, "y_min": 0.0, "y_max": 1e6}
ROI_B = {"x_min": 400.0, "x_max": 440.0, "y_min": 0.0, "y_max": 1e6}


@pytest.fixture
def window(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    win = MainWindow(settings=settings)
    qtbot.addWidget(win)
    yield win
    win.runner.cancel_all()
    win.runner.wait(10000)


@pytest.fixture
def tab(window, qtbot):
    tab = window.tab_widgets[0]
    tab.trace_form.set_value("bintime", 1.0)
    window.ctx.load_file("ptu", PULSE)
    qtbot.waitUntil(lambda: tab.trace is not None and not window.runner.busy, timeout=60000)
    return tab


def _wait(window, qtbot, condition, timeout=60000):
    qtbot.waitUntil(lambda: condition() and not window.runner.busy, timeout=timeout)


def _combo_texts(form, key):
    combo = form.fields[key]
    return [combo.itemText(i) for i in range(combo.count())]


def _components(tab):
    """``{(row label, column title): text}`` of the component table."""
    table = tab.component_table
    titles = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
    return {(table.item(r, 0).text(), titles[c]): table.item(r, c).text()
            for r in range(table.rowCount()) for c in range(1, table.columnCount())}


def test_time_window_between_two_triggers(window, qtbot, tab):
    times = tab.trace.trigger_times
    assert _combo_texts(tab.trace_form, "start_trigger") == [
        text for _, text in model.trigger_options(times)]
    assert not tab.trace_form.fields["start_trigger"].isVisibleTo(tab)

    tab.trace_form.set_values({"start_kind": "trigger", "start_trigger": 1,
                               "end_kind": "trigger", "end_trigger": 2})
    assert tab.trace_form.fields["start_trigger"].isVisibleTo(tab)
    assert not tab.trace_form.fields["tstart"].isVisibleTo(tab)
    tab.update_trace()
    _wait(window, qtbot, lambda: tab.trace.params.start_trigger == 1)
    assert tab.trace.window == (pytest.approx(times[0]), pytest.approx(times[1]))
    assert tab.trace.signal.x[-1] < times[1]
    # The choice survives reloading the file (drop-downs are empty meanwhile).
    window.ctx.load_file("ptu", PULSE)
    _wait(window, qtbot, lambda: tab.trace is not None)
    assert tab.trace.params.start_trigger == 1 and tab.trace.params.end_trigger == 2


def test_rois_feed_the_fit_sources_and_each_roi_is_fitted(window, qtbot, tab, tmp_path,
                                                          monkeypatch):
    assert "ROI 1" not in _combo_texts(tab.fit_form, "source")
    tab.set_rois([ROI_A, ROI_B])
    sources = _combo_texts(tab.fit_form, "source")
    assert {"ROI 1", "ROI 2", "All ROIs combined", "Each ROI separately"} <= set(sources)
    assert _combo_texts(tab.lt_form, "region") == ["Whole time window", "ROI 1", "ROI 2"]

    tab.fit_form.set_values({"source": "each", "n_exp": 1})
    tab.run_fit()
    _wait(window, qtbot, lambda: len(tab.fits) == 2)
    assert [r.roi_index for r in tab.fits] == [0, 1]
    headers = [tab.fit_table.horizontalHeaderItem(c).text()
               for c in range(tab.fit_table.columnCount())]
    assert headers == ["Parameter", "ROI 1", "ROI 2"]
    rows = [tab.fit_table.item(r, 0).text() for r in range(tab.fit_table.rowCount())]
    assert "χ²ᵣ (Poisson)" in rows and "R²" in rows
    # the component table groups its rows per ROI, in the ROI colours
    cells = _components(tab)
    assert ("ROI 1 · 1", "τ (ns)") in cells and ("ROI 2 · Mean τ_int", "τ (ns)") in cells
    label = tab.component_table.item(0, 0)
    assert label.text() == "ROI 1 · 1" and label.foreground().color().name() == "#1f77b4"
    # histogram + residual panel
    assert len(tab.hist_plot.figure.axes) == 2

    target = tmp_path / "fits.csv"
    monkeypatch.setattr(window.ctx, "save_file_dialog", lambda *a, **k: str(target))
    tab.hist_plot.export_csv_dialog()
    assert target.read_text(encoding="utf-8").startswith("# analysis: PTU lifetime fits per ROI")

    # Removing an ROI that is selected as fit source falls back to the whole trace.
    tab.fit_form.set_value("source", "roi:2")
    tab.roi_list.setCurrentRow(1)
    tab.roi_editor.remove_selected()
    assert tab.fit_params().source == "all"


def test_histogram_is_exported_before_a_fit(window, tab, tmp_path, monkeypatch):
    assert not tab.fits
    target = tmp_path / "hist.csv"
    monkeypatch.setattr(window.ctx, "save_file_dialog", lambda *a, **k: str(target))
    payload = tab._histogram_payload()
    assert payload.default_name == f"{PULSE.stem}_lifetime"
    tab.hist_plot.export_csv_dialog()
    assert target.read_text(encoding="utf-8").startswith("# analysis: PTU lifetime histogram")


def test_single_fit_shows_quality_in_title(window, qtbot, tab):
    tab.fit_form.set_value("n_exp", 1)
    tab.run_fit()
    _wait(window, qtbot, lambda: tab.fit is not None)
    title = tab.hist_plot.figure.axes[0].get_title()
    assert "χ²ᵣ" in title and "R²" in title
    assert len(tab.fit.fit["quality"]["weighted_residuals"]) == len(tab.fit.fit["t_fit"])


def test_adding_a_component_is_judged_by_bic(window, qtbot, tab):
    messages = []
    window.ctx.log.message.connect(lambda text, level: messages.append((text, level)))
    tab.fit_form.set_value("n_exp", 1)
    tab.run_fit()
    _wait(window, qtbot, lambda: tab.fit is not None)
    first = tab.fit
    assert first.extra["bic_comparison"].verdict in ("first", "not comparable")

    tab.fit_form.set_value("n_exp", 2)
    tab.run_fit()
    _wait(window, qtbot, lambda: tab.fit is not first)
    comparison = tab.fit.extra["bic_comparison"]
    assert comparison.previous_n_exp == 1 and comparison.verdict == "better"
    assert comparison.delta < -10
    rows = {tab.fit_table.item(r, 0).text(): r for r in range(tab.fit_table.rowCount())}
    assert "BIC" in rows and ("1", "A fitted (counts)") in _components(tab)
    item = tab.fit_table.item(rows["BIC"], 1)
    assert "Δ" in item.text() and item.foreground().color().name() == "#2e7d32"
    assert any("ΔBIC" in text and level == "good" for text, level in messages)
    # the coloured BIC line above the title
    texts = [t.get_text() for t in tab.hist_plot.figure.axes[0].texts]
    assert any("ΔBIC" in t for t in texts)

    # a different fit window is not comparable
    tab.fit_form.set_values({"mode": "manual", "start": 0.0, "end": 40.0})
    tab.run_fit()
    _wait(window, qtbot, lambda: tab.fit.params.mode == "manual")
    assert tab.fit.extra["bic_comparison"].verdict == "not comparable"


def test_fit_warnings_are_logged(window, qtbot, tab):
    messages = []
    window.ctx.log.message.connect(lambda text, level: messages.append((text, level)))
    options = model.FitOptions(min_fraction=0.99)     # every component looks collapsed
    tab.set_fit_options(options)
    tab.fit_form.set_value("n_exp", 2)
    tab.run_fit()
    _wait(window, qtbot, lambda: tab.fit is not None)
    assert tab.fit.fit["warnings"]
    assert any("collapsed" in text and level == "warning" for text, level in messages)


def test_flid_region_is_independent_of_the_lifetime_trace_region(window, qtbot, tab):
    tab.set_rois([ROI_A, ROI_B])
    assert _combo_texts(tab.flid_form, "region") == [
        "Each ROI (one map per ROI)", "Whole time window", "ROI 1", "ROI 2"]
    tab.trace_form.set_value("bintime", 5.0)
    tab.update_trace()
    _wait(window, qtbot, lambda: tab.trace.params.bintime == 5.0)
    tab.flid_form.set_values({"lifetime_bins": 10, "intensity_bins": 10})

    tab.lt_form.set_value("region", "roi:1")    # must not affect the FLID
    tab.flid_form.set_value("region", "window")
    tab.run_flids()
    _wait(window, qtbot, lambda: tab.flid_result is not None, timeout=120000)
    assert len(tab.flid_result.entries) == 1
    assert tab.export_roi.itemText(0) == "Whole time window"

    tab.flid_form.set_value("region", "roi:2")
    tab.run_flids()
    _wait(window, qtbot, lambda: tab.flid_result.params.region == "roi", timeout=120000)
    assert [e.index for e in tab.flid_result.entries] == [1]


def test_roi_lifetime_trace(window, qtbot, tab):
    tab.set_rois([ROI_A])
    tab.lt_form.set_value("region", "roi:1")
    assert not tab.lt_form.fields["bintime"].isVisibleTo(tab)
    tab.run_lifetime_trace()
    _wait(window, qtbot, lambda: tab.lifetime is not None, timeout=120000)
    signal, *_rest, lt, _options, rois = tab.lifetime
    assert lt.roi == 0 and signal.meta["bintime"] == 1.0 and rois == [ROI_A]
    assert tab.plots.currentWidget() is tab.lt_plot


def test_advanced_fit_settings_dialog_round_trip(window, qtbot, tab):
    from kpfm_pl.gui.fit_options import FitOptionsDialog, describe_options

    options = model.FitOptions()
    options.tau_min[0], options.tau_max[0] = 2.0, 2.5
    options.weighting = "poisson"
    options.bin_weighting = "mle"
    options.constrain_sum_A = True
    dialog = FitOptionsDialog(options)
    qtbot.addWidget(dialog)
    assert dialog.form.value("weighting") == "poisson"
    assert dialog.form.value("bin_weighting") == "mle"
    back = dialog.options()
    assert back.tau_min[0] == 2.0 and back.tau_max[0] == 2.5 and back.tau_min[1] is None
    assert back.weighting == "poisson" and back.bin_weighting == "mle" and back.constrain_sum_A
    summary = describe_options(back)
    assert "lifetime fit: Poisson least squares" in summary and "trace/FLID: Poisson MLE" in summary
    assert describe_options(model.FitOptions()) == "defaults"
    assert "sum(A)" in describe_options(back)
    # fractions are entered in percent and come back as fractions
    tuned = model.FitOptions(bic_threshold=6.0, min_fraction=0.02, merge_rel=0.05,
                             weighting="mle", mean_lifetime="amplitude",
                             decay_correction=False)
    tuned_dialog = FitOptionsDialog(tuned)
    qtbot.addWidget(tuned_dialog)
    assert tuned_dialog.form.value("min_fraction") == pytest.approx(2.0)
    again = tuned_dialog.options()
    assert again.bic_threshold == 6.0 and again.weighting == "mle"
    assert again.mean_lifetime == "amplitude" and not again.decay_correction
    assert "amplitude-weighted" in describe_options(again)
    assert "no incomplete-decay correction" in describe_options(again)
    assert again.min_fraction == pytest.approx(0.02) and again.merge_rel == pytest.approx(0.05)
    assert "|ΔBIC| threshold 6" in describe_options(again)
    # an invalid entry keeps the dialog open with a message
    dialog.cells[("tau_max", 0)].set_value(1.0)
    dialog._accept()
    assert dialog.error_label.text() and dialog.result() == 0

    tab.set_fit_options(back)
    assert describe_options(back) in tab.options_label.text()
    tab.fit_form.set_value("n_exp", 1)
    tab.run_fit()
    _wait(window, qtbot, lambda: tab.fit is not None)
    assert 2.0 <= tab.fit.fit["tau"][0] <= 2.5
    assert tab.fit.fit["weighting"] == "poisson"
    rows = {tab.fit_table.item(r, 0).text(): r for r in range(tab.fit_table.rowCount())}
    assert tab.fit_table.item(rows["Fit statistic"], 1).text() == "Poisson least squares"


def test_mle_fallback_is_logged_every_time(window, qtbot, tab, monkeypatch):
    from kpfm_pl import fitting

    messages = []
    window.ctx.log.message.connect(lambda text, level: messages.append((text, level)))
    monkeypatch.setattr(fitting, "_MLE_MAX_ITER", 0)        # no MLE can converge
    tab.fit_form.set_value("n_exp", 1)
    for attempt in range(2):                                # not swallowed the second time
        previous = tab.fit
        tab.run_fit()
        _wait(window, qtbot, lambda: tab.fit is not previous)
        logged = [t for t, level in messages if "did not converge" in t and level == "warning"]
        assert len(logged) == attempt + 1
    assert tab.fit.fit["weighting"] == "poisson" and tab.fit.fit["mle_converged"] is False
    rows = {tab.fit_table.item(r, 0).text(): r for r in range(tab.fit_table.rowCount())}
    assert "MLE did not converge" in tab.fit_table.item(rows["Fit statistic"], 1).text()
    assert "MLE did not converge" in tab.hist_plot.figure.axes[0].get_title()


def test_lifetime_shift_is_applied_per_file(window, qtbot, tab):
    assert tab.shift_value() is None and tab.trace.lifetime_shift_ns == 0.0
    assert not tab.shift_form.fields["shift_ns"].isVisibleTo(tab)
    tab.shift_form.set_value("kind", "auto")
    tab.update_trace()
    _wait(window, qtbot, lambda: tab.trace.lifetime_shift_ns > 0)
    shift = tab.trace.lifetime_shift_ns
    assert f"{shift:.3f} ns" in tab.shift_label.text()
    hist = window.ctx.state.ptu.reader.lifetime_histogram()
    assert hist.x[np.argmax(hist.y)] * 1e9 == pytest.approx(5.0, abs=0.5)

    tab.fit_form.set_value("n_exp", 1)
    tab.run_fit()
    _wait(window, qtbot, lambda: tab.fit is not None)
    rows = [tab.fit_table.item(r, 0).text() for r in range(tab.fit_table.rowCount())]
    assert "Lifetime shift (ns)" in rows
    assert "shifted by" in tab.hist_plot.figure.axes[0].get_title()

    tab.shift_form.set_values({"kind": "manual", "shift_ns": 10.0})
    assert tab.shift_form.fields["shift_ns"].isVisibleTo(tab)
    tab.update_trace()
    _wait(window, qtbot, lambda: tab.trace.lifetime_shift_ns != shift)
    assert tab.trace.lifetime_shift_ns == pytest.approx(10.0, abs=0.1)

    # A newly loaded file starts unshifted, and so does the control.
    window.ctx.load_file("ptu", PULSE)
    _wait(window, qtbot, lambda: tab.trace is not None)
    assert tab.shift_value() is None and tab.trace.lifetime_shift_ns == 0.0
    assert tab.shift_label.text() == ""


def test_groups_fold(tab):
    group = tab.groups["FLID"]
    assert not group.is_expanded() and not group.body.isVisibleTo(tab)
    group.button.click()
    assert group.is_expanded() and group.body.isVisibleTo(tab)
    assert np.all([tab.groups[t].is_expanded() for t in ("Time trace", "Lifetime fit")])


def test_fit_table_and_title_show_tau_int_with_error(window, qtbot, tab):
    tab.fit_form.set_value("n_exp", 2)
    tab.run_fit()
    _wait(window, qtbot, lambda: tab.fit is not None)
    # one row per component, then the mean lifetime in the τ column
    table = tab.component_table
    assert [table.item(r, 0).text() for r in range(table.rowCount())] == ["1", "2", "Mean τ_int"]
    titles = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
    assert titles == ["Component", "τ (ns)", "A (%)", "I (%)", "A (counts)", "A fitted (counts)"]
    cells = _components(tab)
    assert "±" in cells[("Mean τ_int", "τ (ns)")] and cells[("Mean τ_int", "A (%)")] == ""
    assert "τ_int = " in tab.hist_plot.figure.axes[0].get_title()
    # amplitude and intensity percentages with errors, counts for reference
    for row, column in (("1", "A (%)"), ("2", "I (%)"), ("1", "A (counts)"), ("1", "τ (ns)")):
        assert "±" in cells[(row, column)]
    assert cells[("1", "A fitted (counts)")]
    # the second table holds no per-component values and no error rows
    rows = [tab.fit_table.item(r, 0).text() for r in range(tab.fit_table.rowCount())]
    assert not any(r.startswith(("τ", "A1", "I1")) or "err" in r for r in rows)
    assert {"Offset", "χ²ᵣ (Poisson)", "BIC", "Fit start (ns)"} <= set(rows)
    assert "normalize" not in tab.fit_form.fields

    # amplitude-weighted mean, no incomplete-decay correction
    tab.set_fit_options(model.FitOptions(mean_lifetime="amplitude", decay_correction=False))
    previous = tab.fit
    tab.run_fit()
    _wait(window, qtbot, lambda: tab.fit is not previous)
    labels = [table.item(r, 0).text() for r in range(table.rowCount())]
    assert labels[-1] == "Mean τ_amp"
    titles = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
    assert "A fitted (counts)" not in titles
    assert "τ_amp = " in tab.hist_plot.figure.axes[0].get_title()


def test_flid_has_its_own_fit_window(window, qtbot, tab):
    tab.fit_form.set_values({"mode": "manual", "start": 1.5})
    assert tab.flid_params().mode == "fraction"   # not taken from 'Lifetime fit'
    tab.copy_fit_window()
    params = tab.flid_params()
    assert params.mode == "manual" and params.start == 1.5
    assert tab.flid_form.fields["start"].isVisibleTo(tab.flid_form)
    tab.flid_form.set_value("method", "mean_arrival")
    assert not tab.flid_form.fields["n_exp"].isVisibleTo(tab.flid_form)
    assert tab.flid_form.fields["truncation_correction"].isVisibleTo(tab.flid_form)


def test_flid_maps_redraw_without_refit_square_and_pop_out(window, qtbot, tab, monkeypatch):
    from kpfm_pl.gui.tabs.flid_window import FlidWindow

    tab.set_rois([ROI_A])
    tab.flid_form.set_values({"lifetime_bins": 10, "intensity_bins": 10, "method": "both",
                              "min_counts": 0})
    tab.run_flids()
    _wait(window, qtbot, lambda: tab.flid_result is not None, timeout=120000)
    result = tab.flid_result
    maps = list(tab._flid_axes)
    assert len(maps) == 2 and all(ax.get_box_aspect() == 1 for ax in maps)
    assert tab.export_content.count() == 3

    # Map settings redraw at once from the stored per-bin results (no job).
    submitted = []
    monkeypatch.setattr(window.runner, "submit", lambda *a, **k: submitted.append(a))
    tab.flid_form.set_value("sigma", 0.0)
    tab.flid_form.set_value("lifetime_bins", 20)
    assert not submitted
    assert tab.flid_result is not result and tab.flid_result.params.sigma == 0.0
    entry = tab.flid_result.entries[0]
    assert entry.traces is result.entries[0].traces
    assert entry.images["fit"].data.shape == (10, 20)
    # Changing what is computed waits for 'Compute FLIDs'.
    tab.flid_form.set_value("n_exp", 2)
    assert tab.flid_result.params.n_exp == 1

    tab.pop_out_selected()
    windows = list(tab._popouts)
    assert len(windows) == 2 and all(isinstance(w, FlidWindow) for w in windows)
    popped = windows[0]
    assert popped.ax.get_box_aspect() == 1
    popped.form.set_values({"scale": "log", "cmap": "magma"})
    assert type(popped.im.norm).__name__ == "LogNorm" and popped.im.get_cmap().name == "magma"
    popped.form.set_values({"scale": "linear", "vmin": 0.0, "vmax": 0.5})
    assert popped.im.norm.vmax == 0.5
    popped.form.set_value("vmax", None)
    popped.form.set_value("vmin", 1e6)
    assert "smaller" in popped.message.text()
    popped.close()
    qtbot.waitUntil(lambda: len(tab._popouts) == 1)

    # Double-clicking a map in the FLID plot pops it out as well.
    from matplotlib.backend_bases import MouseEvent

    tab.flid_plot.canvas.draw()
    ax = next(iter(tab._flid_axes))
    x, y = ax.transAxes.transform((0.5, 0.5))
    MouseEvent("button_press_event", tab.flid_plot.canvas, x, y, button=1,
               dblclick=True)._process()
    assert len(tab._popouts) == 2


def test_irf_file_reconvolution_and_per_bin_reuse(window, qtbot, tab, tmp_path, monkeypatch):
    messages = []
    window.ctx.log.message.connect(lambda text, level: messages.append((text, level)))
    assert tab.irf is None and not tab.irf_form.fields["use"].isEnabled()
    # The test data hold no IRF recording; the decay itself exercises the code paths.
    monkeypatch.setattr(window.ctx, "open_file_dialog", lambda *a, **k: str(PULSE))
    tab.choose_irf()
    _wait(window, qtbot, lambda: tab.irf is not None)
    assert tab.irf_form.value("use") and PULSE.name in tab.irf_label.text()
    assert window.ctx.settings.value("ptu/irf_path") == str(PULSE)

    # per-bin fits without an IRF lifetime fit fall back to the tail fit with a warning
    tab.lt_form.set_values({"bintime": 60.0, "n_exp": 1})
    tab.run_lifetime_trace()
    _wait(window, qtbot, lambda: tab.lifetime is not None, timeout=120000)
    assert not tab.lifetime[0].meta["reconvolution"]
    assert any("IRF not used" in text and level == "warning" for text, level in messages)

    tab.irf_form.set_value("fit_background", True)
    tab.fit_form.set_value("n_exp", 1)
    tab.run_fit()
    _wait(window, qtbot, lambda: tab.fit is not None)
    assert tab.fit.fit["reconvolution"] and tab.fit.fit["fit_background"]
    rows = [tab.fit_table.item(r, 0).text() for r in range(tab.fit_table.rowCount())]
    assert "IRF shift δ (ns)" in rows and "IRF FWHM (ns)" in rows
    labels = [line.get_label() for line in tab.hist_plot.figure.axes[0].get_lines()]
    assert "IRF (shifted, scaled)" in labels

    previous = tab.lifetime
    tab.run_lifetime_trace()
    _wait(window, qtbot, lambda: tab.lifetime is not previous, timeout=120000)
    assert tab.lifetime[0].meta["reconvolution"]
    assert tab.lifetime[0].meta["irf_shift_ns"] == pytest.approx(tab.fit.fit["shift_ns"])

    # remembered: a new window with the same settings loads the IRF again
    second = MainWindow(settings=window.ctx.settings)
    qtbot.addWidget(second)
    other = second.tab_widgets[0]
    qtbot.waitUntil(lambda: other.irf is not None and not second.runner.busy, timeout=60000)
    assert other.irf_form.value("use")
    other.clear_irf()
    assert other.irf is None and not other.irf_form.value("use")
    assert not window.ctx.settings.value("ptu/irf_path")
    second.runner.wait(10000)


def test_fitted_background_and_prerise_without_irf(window, qtbot, tab):
    messages = []
    window.ctx.log.message.connect(lambda text, level: messages.append((text, level)))
    # both are off by default; the background checkbox does not need an IRF
    assert not tab.fit_form.value("prerise") and not tab.irf_form.value("fit_background")
    assert tab.irf_form.fields["fit_background"].isVisibleTo(tab.irf_form)
    assert not tab.irf_form.fields["fit_scatter"].isVisibleTo(tab.irf_form)
    assert not tab.fit_form.fields["prerise_gap"].isVisibleTo(tab.fit_form)
    tab.fit_form.set_value("prerise", True)
    assert tab.fit_form.fields["prerise_gap"].isVisibleTo(tab.fit_form)
    assert tab.fit_params().prerise and tab.fit_params().prerise_gap == 2.0

    # per-bin fits without a lifetime fit with fitted background: dark counts + warning
    tab.irf_form.set_value("fit_background", True)
    assert "reuse the background" in tab.irf_note.text()
    tab.lt_form.set_values({"bintime": 60.0, "n_exp": 1})
    tab.run_lifetime_trace()
    _wait(window, qtbot, lambda: tab.lifetime is not None, timeout=120000)
    assert tab.lifetime[0].meta["background_rate"] is None and tab.lifetime[0].meta["prerise"]
    assert any("Fitted background not used" in text and level == "warning"
               for text, level in messages)

    tab.run_fit()
    _wait(window, qtbot, lambda: tab.fit is not None)
    fit = tab.fit.fit
    assert fit["fit_background"] and fit["prerise"] and not fit.get("reconvolution")
    rows = [tab.fit_table.item(r, 0).text() for r in range(tab.fit_table.rowCount())]
    assert "Bins before the rise" in rows
    offset = tab.fit_table.item(rows.index("Offset"), 1).text()
    assert "±" in offset                              # fitted: shown with its error
    ax = tab.hist_plot.figure.axes[0]
    labels = [patch.get_label() for patch in ax.patches]
    assert "Fit window" in labels and "Before the rise (fitted)" in labels
    # the fitted curve is broken at the rising edge
    curve = next(line for line in ax.get_lines() if line.get_label() == "Fit")
    assert np.isnan(curve.get_xdata()[fit["prerise_points"]])

    previous = tab.lifetime
    tab.run_lifetime_trace()
    _wait(window, qtbot, lambda: tab.lifetime is not previous, timeout=120000)
    duration = tab.fit.histogram.meta["acquisition_duration"]
    assert tab.lifetime[0].meta["background_rate"] == pytest.approx(fit["background"] / duration)


def test_flid_prerise_setting_is_copied_and_passed_on(window, qtbot, tab):
    form = tab.flid_form
    assert not tab.flid_params().prerise
    assert not form.fields["prerise_gap"].isVisibleTo(form)
    tab.fit_form.set_values({"prerise": True, "prerise_gap": 3.5})
    assert not tab.flid_params().prerise           # the FLID keeps its own fit window
    tab.copy_fit_window()
    params = tab.flid_params()
    assert params.prerise and params.prerise_gap == 3.5
    assert params.prerise_kwargs() == {"prerise": True, "prerise_gap": 3.5}
    assert form.fields["prerise_gap"].isVisibleTo(form)
    form.set_value("method", "mean_arrival")       # no fit: the setting does not apply
    assert not form.fields["prerise"].isVisibleTo(form)


def test_period_fraction_in_the_advanced_settings(qtbot):
    from kpfm_pl.gui.fit_options import FitOptionsDialog, describe_options

    dialog = FitOptionsDialog(model.FitOptions(period_fraction=0.8))
    qtbot.addWidget(dialog)
    assert dialog.form.value("period_fraction") == pytest.approx(80.0)   # shown in percent
    assert dialog.options().period_fraction == pytest.approx(0.8)
    assert "custom warning thresholds" in describe_options(dialog.options())
    assert describe_options(model.FitOptions()) != describe_options(dialog.options())


def test_seam_bins_tickbox_marks_and_keeps_the_bins(window, qtbot, tab):
    assert tab.fit_params().exclude_seam and tab.flid_params().exclude_seam   # on by default
    tab.run_fit()
    _wait(window, qtbot, lambda: tab.fit is not None)
    fit = tab.fit.fit
    assert fit["seam_points"] == 1
    labels = tab.hist_plot.figure.axes[0].get_legend_handles_labels()[1]
    assert "Seam bins, not fitted (1)" in labels
    rows = [tab.fit_table.item(r, 0).text() for r in range(tab.fit_table.rowCount())]
    assert "Seam bins (not fitted)" in rows

    tab.fit_form.set_value("exclude_seam", False)
    assert tab.flid_params().exclude_seam          # the FLID keeps its own fit window
    tab.copy_fit_window()
    assert not tab.flid_params().exclude_seam
    previous = tab.fit
    tab.run_fit()
    _wait(window, qtbot, lambda: tab.fit is not previous)
    assert tab.fit.fit["seam_points"] == 0
    labels = tab.hist_plot.figure.axes[0].get_legend_handles_labels()[1]
    assert not any(label.startswith("Seam bins") for label in labels)

