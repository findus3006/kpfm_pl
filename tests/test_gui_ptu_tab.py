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
    dialog = FitOptionsDialog(options)
    qtbot.addWidget(dialog)
    back = dialog.options()
    assert back.tau_min[0] == 2.0 and back.tau_max[0] == 2.5 and back.tau_min[1] is None
    assert back.weighting == "poisson"
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
