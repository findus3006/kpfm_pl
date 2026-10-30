"""Tests of the GUI's FLID workflow: Qt-free model and offscreen tab."""

from pathlib import Path

import numpy as np
import pytest

from kpfm_pl.gui.export import write_payload_csv
from kpfm_pl.gui.models import flid as model
from kpfm_pl.gui.models.files import load_ptu
from kpfm_pl.gui.models.ptu import TraceParams, compute_trace

DATA_DIR = Path(__file__).parents[1] / "examples" / "Test data"
PULSE = DATA_DIR / "ptu_test2_LaserPulse(DR1_800nW_1200_ChargeAccu_000).ptu"

pytestmark = pytest.mark.skipif(not PULSE.exists(), reason="PTU test file missing")

# Short windows and coarse bins keep the per-bin fits fast.
ROI_A = {"x_min": 300.0, "x_max": 340.0, "y_min": 0.0, "y_max": 1e5}
ROI_B = {"x_min": 400.0, "x_max": 440.0, "y_min": 0.0, "y_max": 1e5}
EMPTY = {"x_min": 0.0, "x_max": 100.0, "y_min": 1e5, "y_max": 2e5}


@pytest.fixture(scope="module")
def handle():
    return load_ptu(PULSE)


def test_default_roi_spans_first_to_last_marker(handle):
    trace = compute_trace(handle, TraceParams(bintime=1.0))
    roi = model.default_roi(trace)
    assert roi["x_min"] == pytest.approx(trace.marker_times[0])
    assert roi["x_max"] == pytest.approx(trace.marker_times[-1])
    assert roi["y_max"] == pytest.approx(trace.signal.y.max() + 1)


def test_scale_rois_scales_count_range_only():
    scaled = model.scale_rois([ROI_A], 2.0)[0]
    assert scaled["x_min"] == ROI_A["x_min"] and scaled["y_max"] == 2e5


def test_lifetime_range_validation():
    assert model.FlidParams(lifetime_max=None).lifetime_range() is None
    with pytest.raises(ValueError):
        model.FlidParams(lifetime_min=5, lifetime_max=5).lifetime_range()


def test_compute_flids_shares_ranges_skips_empty_and_reports_progress(handle, tmp_path):
    params = model.FlidParams(lifetime_bins=20, intensity_bins=20)
    calls = []
    result = model.compute_flids(handle, TraceParams(bintime=1.0), params,
                                 [ROI_A, EMPTY, ROI_B],
                                 progress=lambda d, t: calls.append((d, t)))
    assert [e.index for e in result.entries] == [0, 2]
    assert [s[0] for s in result.skipped] == [1]
    assert not result.fallback
    ranges = {e.image.meta["intensity_range"] for e in result.entries}
    assert ranges == {result.intensity_range}
    assert result.entries[0].image.meta["lifetime_range"] == (0.0, 20.0)
    assert all(e.n_fitted > 0 for e in result.entries)
    # One overall, monotonic progress scale ending at done == total.
    assert calls[-1][0] == calls[-1][1]
    assert [d for d, _ in calls] == sorted(d for d, _ in calls)

    entry = result.entries[1]
    path = write_payload_csv(tmp_path / "map.csv", model.flid_payload(result, entry))
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# analysis: FLID map")
    assert '"roi_number": 3' in text
    path = write_payload_csv(tmp_path / "bins.csv", model.bin_fits_payload(result, entry))
    header = next(line for line in path.read_text().splitlines() if not line.startswith("#"))
    assert header.startswith("x (s),counts y") and "tau1 y (ns)" in header


def test_flid_region_selects_window_or_single_roi(handle, tmp_path):
    small = dict(lifetime_bins=10, intensity_bins=10)
    trace_params = TraceParams(bintime=5.0)
    window = model.compute_flids(handle, trace_params,
                                 model.FlidParams(region="window", **small), [ROI_A, ROI_B])
    assert len(window.entries) == 1 and not window.fallback
    entry = window.entries[0]
    assert window.entry_name(entry) == "Whole time window"
    trace = compute_trace(handle, trace_params)
    assert entry.roi == model.window_roi(trace)
    assert entry.roi["x_min"] == trace.signal.x[0]
    meta = model.flid_payload(window, entry).metadata["results"]
    assert meta["region"] == "Whole time window" and meta["roi_number"] is None

    single = model.compute_flids(handle, trace_params,
                                 model.FlidParams(region="roi", roi=1, **small), [ROI_A, ROI_B])
    assert [e.index for e in single.entries] == [1]
    assert single.entry_name(single.entries[0]) == "ROI 2"
    with pytest.raises(ValueError, match="ROI 3"):
        model.compute_flids(handle, trace_params, model.FlidParams(region="roi", roi=2),
                            [ROI_A, ROI_B])


def test_compute_flids_without_rois_uses_fallback(handle):
    result = model.compute_flids(handle, TraceParams(bintime=5.0),
                                 model.FlidParams(lifetime_bins=10, intensity_bins=10), [])
    assert result.fallback and len(result.entries) == 1


def test_compute_flids_cancellation_propagates(handle):
    class Stop(Exception):
        pass

    def cancel(done, total):
        if done > 0:
            raise Stop

    with pytest.raises(Stop):
        model.compute_flids(handle, TraceParams(bintime=1.0), model.FlidParams(),
                            [ROI_A, ROI_B], progress=cancel)


# ---------------------------------------------------------------- Qt tab
@pytest.fixture
def window(qtbot, tmp_path):
    pytest.importorskip("pytestqt")
    from PySide6.QtCore import QSettings

    from kpfm_pl.gui.app import MainWindow

    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    win = MainWindow(settings=settings)
    qtbot.addWidget(win)
    yield win
    win.runner.cancel_all()
    win.runner.wait(10000)


def _flid_tab(window):
    """The FLID workflow lives in the merged PL lifetime tab."""
    from kpfm_pl.gui.tabs.ptu_tab import PTUTab

    return next(t for t in window.tab_widgets if isinstance(t, PTUTab))


def _load(window, qtbot, tab):
    tab.trace_form.set_value("bintime", 1.0)
    window.ctx.load_file("ptu", PULSE)
    qtbot.waitUntil(lambda: tab.trace is not None and not window.runner.busy, timeout=60000)


def _drag(canvas, ax, start, stop):
    from matplotlib.backend_bases import MouseEvent

    for name, xy in (("button_press_event", start), ("motion_notify_event", stop),
                     ("button_release_event", stop)):
        x, y = ax.transData.transform(xy)
        MouseEvent(name, canvas, x, y, button=1)._process()


def test_drawn_box_appears_in_roi_list_and_survives_redraw(window, qtbot):
    tab = _flid_tab(window)
    _load(window, qtbot, tab)
    ax = tab.trace_plot.figure.axes[0]
    _drag(tab.trace_plot.canvas, ax, (300.0, 100.0), (340.0, 1500.0))
    assert tab.roi_list.count() == 1
    roi = tab.rois[0]
    assert roi["x_min"] == pytest.approx(300.0) and roi["y_max"] == pytest.approx(1500.0)

    # Doubling the bin time keeps the ROI, with its count range doubled.
    tab.trace_form.set_value("bintime", 2.0)
    tab.update_trace()
    qtbot.waitUntil(lambda: tab.trace.params.bintime == 2.0 and not window.runner.busy,
                    timeout=60000)
    selector = tab.roi_editor.selector
    assert len(selector.boxes) == 1
    assert tab.rois[0]["y_max"] == pytest.approx(3000.0)

    tab.roi_list.setCurrentRow(0)
    tab.roi_editor.remove_selected()
    assert tab.rois == [] and tab.roi_editor.selector.boxes == []


def test_flids_computed_plotted_and_exported(window, qtbot, tmp_path, monkeypatch):
    tab = _flid_tab(window)
    _load(window, qtbot, tab)
    tab.set_rois([ROI_A, ROI_B])
    tab.flid_form.set_value("lifetime_bins", 20)
    tab.flid_form.set_value("intensity_bins", 20)
    tab.run_flids()
    qtbot.waitUntil(lambda: tab.flid_result is not None and not window.runner.busy,
                    timeout=120000)
    assert len(tab.flid_result.entries) == 2
    assert tab.flid_result.trace.params.bintime == 1.0   # the trace's bin time
    assert len(tab.flid_plot.figure.subfigs) == 2
    assert tab.export_roi.count() == 2

    tab.export_roi.setCurrentIndex(1)
    target = tmp_path / "flid.csv"
    monkeypatch.setattr(window.ctx, "save_file_dialog", lambda *a, **k: str(target))
    tab.flid_plot.export_csv_dialog()
    assert '"roi_number": 2' in target.read_text(encoding="utf-8")

    png = tmp_path / "flid.png"
    monkeypatch.setattr(window.ctx, "save_file_dialog", lambda *a, **k: str(png))
    tab.flid_plot.save_figure_dialog()
    assert png.stat().st_size > 0
    assert np.isfinite(tab.flid_result.intensity_range).all()


# ------------------------------------------- re-histogram, methods, progress
def test_remap_rebuilds_maps_without_refitting(handle):
    params = model.FlidParams(lifetime_bins=20, intensity_bins=20, min_counts=0)
    result = model.compute_flids(handle, TraceParams(bintime=1.0), params, [ROI_A])
    entry = result.entries[0]
    remapped = model.remap_flids(result, model.FlidParams(lifetime_bins=10, intensity_bins=20,
                                                          sigma=0.0, min_counts=0))
    new = remapped.entries[0]
    assert new.traces["fit"] is entry.traces["fit"]       # per-bin fits reused
    assert new.image.data.shape == (20, 10) and entry.image.data.shape == (20, 20)
    assert new.image.data.sum() == pytest.approx(entry.n_fitted)  # sigma 0: raw counts
    with pytest.raises(ValueError, match="map settings"):
        model.remap_flids(result, model.FlidParams(n_exp=2, min_counts=0))
    assert params.same_lifetimes(model.FlidParams(sigma=5, lifetime_max=None, min_counts=0))


def test_both_methods_give_comparable_maps_and_progress_counts_bins(handle, tmp_path):
    params = model.FlidParams(method="both", lifetime_bins=20, intensity_bins=20,
                              mode="manual", start=0.0, min_counts=0)
    calls = []
    result = model.compute_flids(handle, TraceParams(bintime=1.0), params, [ROI_A, EMPTY],
                                 progress=lambda d, t: calls.append((d, t)))
    entry = result.entries[0]
    assert entry.methods == ["fit", "mean_arrival"]
    fit, arrival = entry.images["fit"], entry.images["mean_arrival"]
    assert fit.meta["lifetime_range"] == arrival.meta["lifetime_range"]
    assert fit.meta["intensity_range"] == arrival.meta["intensity_range"]
    # The test decay is multi-exponential: a single-exponential fit follows the
    # short component, the mean arrival time the (longer) intensity-weighted mean.
    tau_fit = np.nanmedian(entry.traces["fit"].y)
    tau_arrival = np.nanmedian(entry.traces["mean_arrival"].y)
    assert 1.0 < tau_fit < tau_arrival < 15.0
    # Progress counts time bins: 40 bins per method for ROI A (+ the skipped ROI).
    assert calls[-1][0] == calls[-1][1]
    assert 80 <= calls[-1][1] < 1000
    header = next(line for line in write_payload_csv(
        tmp_path / "bins.csv", model.bin_fits_payload(result, entry)).read_text().splitlines()
        if not line.startswith("#"))
    assert "tau1 y (ns)" in header and "tau_mean_arrival y (ns)" in header
    payload = model.flid_payload(result, entry, "mean_arrival")
    assert payload.default_name.endswith("_mean_arrival")
    assert payload.metadata["results"]["method"] == "mean_arrival"


def test_auto_bintime_per_roi_scales_the_counts_axis(handle):
    params = model.FlidParams(method="mean_arrival", lifetime_bins=10, intensity_bins=10,
                              min_counts=1e4, coverage=0.5, auto_bintime=True)
    result = model.compute_flids(handle, TraceParams(bintime=1.0), params, [ROI_A])
    entry = result.entries[0]
    assert entry.bintime > 1.0
    factor = entry.bintime / 1.0
    assert entry.image.meta["intensity_range"] == pytest.approx(
        tuple(v * factor for v in result.intensity_range))
    assert entry.image.units["y"] == f"counts/{entry.bintime * 1e3:g} ms"


def test_count_check_text_reports_failed_checks_only():
    check = {"passed": False, "fraction": 0.4, "bintime": 0.5, "min_counts": 100.0,
             "coverage": 0.9, "suggested_bintime": 2.0}
    text = model.count_check_text(check)
    assert "40%" in text and "at least 2 s" in text
    assert model.count_check_text({**check, "passed": True}) is None
    assert model.count_check_text(None) is None
