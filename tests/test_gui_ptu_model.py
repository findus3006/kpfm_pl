"""Qt-free tests of the GUI's PTU workflow model."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from kpfm_pl.core import Signal
from kpfm_pl.gui.export import write_payload_csv
from kpfm_pl.gui.models import ptu as model
from kpfm_pl.gui.models.files import load_ptu

DATA_DIR = Path(__file__).parents[1] / "examples" / "Test data"
PULSE = DATA_DIR / "ptu_test2_LaserPulse(DR1_800nW_1200_ChargeAccu_000).ptu"
NO_PULSE = DATA_DIR / "ptu_test1_NoLaserPulse_(360s_400nW_532nm_000).ptu"

pytestmark = pytest.mark.skipif(not PULSE.exists(), reason="PTU test file missing")


@pytest.fixture(scope="module")
def handle():
    return load_ptu(PULSE)


def test_trace_reports_hardware_markers(handle):
    result = model.compute_trace(handle, model.TraceParams(bintime=0.5))
    assert len(result.signal) > 0
    assert result.marker_times is not None and len(result.marker_times) > 0
    assert len(result.triggers) == len(result.marker_times)


def test_legacy_trace_uses_edge_triggers():
    legacy = load_ptu(NO_PULSE)
    result = model.compute_trace(legacy, model.TraceParams(bintime=0.5, laser_pulse=False))
    assert result.marker_times is not None and len(result.marker_times) == 2


def test_states_use_current_trace_parameters_after_a_windowed_call(handle):
    params = model.TraceParams(bintime=0.5)
    full = model.compute_states(handle, params, model.StateParams(off=0.0, on=1.0))
    # A windowed trace replaces the PTU's cached trace ...
    handle.reader.time_trace(bintime=0.05, tstart=0, tend=10)
    again = model.compute_states(handle, params, model.StateParams(off=0.0, on=1.0))
    # ... but the model rebuilds it, so the result is unchanged.
    np.testing.assert_array_equal(full["on"].y, again["on"].y)


def test_fit_and_summary(handle):
    result = model.compute_fit(handle, model.TraceParams(bintime=0.5, dark_counts=2.25),
                               model.FitParams(n_exp=2), model.StateParams())
    summary = model.fit_summary(result.fit)
    assert {"A1", "tau1_ns", "A2", "tau2_ns", "r_squared"} <= set(summary)
    assert summary["success"]


def test_fit_rejects_empty_state(handle):
    with pytest.raises(ValueError, match="no photons"):
        model.compute_fit(handle, model.TraceParams(bintime=0.5),
                          model.FitParams(source="off"), model.StateParams(off=0.0, on=0.5))


def test_lifetime_trace_reports_progress_and_exports(handle, tmp_path):
    params = model.TraceParams(bintime=0.5, tstart=0.0, tend=60.0)
    fit, lt = model.FitParams(), model.LifetimeTraceParams(bintime=5.0, n_exp=1)
    calls = []
    signal = model.compute_lifetime_trace(handle, params, fit, lt,
                                          progress=lambda d, t: calls.append((d, t)))
    assert calls and calls[-1][0] == calls[-1][1]
    payload = model.lifetime_trace_payload(signal, str(PULSE), params, fit, lt)
    header = write_payload_csv(tmp_path / "lt.csv", payload).read_text().splitlines()
    assert header[0] == "# analysis: PTU lifetime trace"
    columns = next(line for line in header if not line.startswith("#"))
    assert columns.startswith("x (s),") and "tau1 y (ns)" in columns


def test_trace_and_fit_payloads(handle, tmp_path):
    params = model.TraceParams(bintime=0.5)
    trace = model.compute_trace(handle, params)
    payload = model.trace_payload(trace)
    assert payload.metadata["parameters"]["time_trace"]["bintime"] == 0.5
    assert payload.metadata["results"]["laser_trigger_times_s"]
    write_payload_csv(tmp_path / "trace.csv", payload)

    fit = model.compute_fit(handle, params, model.FitParams(n_exp=1), model.StateParams())
    payload = model.fit_payload(fit)
    assert "tau1_ns" in payload.metadata["results"]
    assert "reduced_chi_squared" in payload.metadata["results"]
    # histogram, fit curve and weighted residuals
    assert [sig.units["y"] for sig in payload.signals][1:] == ["counts", "sigma"]
    write_payload_csv(tmp_path / "fit.csv", payload)


def test_histogram_can_be_exported_without_a_fit(handle, tmp_path):
    params, states = model.TraceParams(bintime=0.5), model.StateParams()
    hist = model.preview_histogram(handle, params, model.FitParams(), states)
    payload = model.histogram_payload(hist, str(handle.path), params, states)
    assert payload.metadata["analysis"] == "PTU lifetime histogram"
    [data] = payload.signals
    assert data.units["x"] == "ns" and data.x[int(np.argmax(data.y))] == 0.0
    text = write_payload_csv(tmp_path / "hist.csv", payload).read_text(encoding="utf-8")
    assert text.startswith("# analysis: PTU lifetime histogram")


def test_export_names_keep_the_full_file_name(handle):
    source = str(Path("C:/data") / "Data1 500nW.ptu")
    params, states = model.TraceParams(bintime=1.0), model.StateParams()
    hist = model.compute_histogram(handle, params, model.FitParams(), states)
    assert model.histogram_payload(hist, source, params, states).default_name ==         "Data1_500nW_lifetime"
    on = model.compute_histogram(handle, params, model.FitParams(source="on"), states)
    assert model.histogram_payload(on, source, params, states).default_name ==         "Data1_500nW_on_lifetime"

    fit = model.compute_fit(handle, params, model.FitParams(n_exp=1), states)
    fit = dataclasses.replace(fit, source=source)
    assert model.fit_payload(fit).default_name == "Data1_500nW_lifetime_fit"
    results, _ = model.compute_roi_fits(handle, params,
                                        model.FitParams(source="each", n_exp=1),
                                        states, [ROI_A, ROI_B])
    results = [dataclasses.replace(r, source=source) for r in results]
    assert model.fit_payload(results[1]).default_name == "Data1_500nW_ROI2_lifetime_fit"
    assert model.fits_payload(results).default_name == "Data1_500nW_ROIs_lifetime_fits"


# ---------------------------------------------------------- trigger windows
def test_trigger_window_resolves_to_marker_times(handle):
    params = model.TraceParams(bintime=0.5)
    with handle.lock:
        times = model.trigger_times(handle, params)
    assert times is not None and len(times) == 2
    assert model.trigger_options(times)[1] == (2, f"#2 – {times[1]:.3f} s")

    windowed = model.TraceParams(bintime=0.5, start_trigger=1, end_trigger=2)
    result = model.compute_trace(handle, windowed)
    assert result.window == (pytest.approx(times[0]), pytest.approx(times[1]))
    x = result.signal.x
    assert x[0] <= times[0] < x[0] + 0.5 and x[-1] < times[1]
    # every trigger of the file stays available for the drop-downs
    np.testing.assert_array_equal(result.trigger_times, times)
    assert result.marker_times is not None and result.marker_times[0] == times[0]

    later = model.compute_trace(handle, model.TraceParams(bintime=0.5, start_trigger=2))
    np.testing.assert_array_equal(later.marker_times, times[1:])


def test_trigger_window_validation():
    times = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="trigger 3"):
        model.resolve_window(model.TraceParams(end_trigger=3), times)
    with pytest.raises(ValueError, match="before it starts"):
        model.resolve_window(model.TraceParams(start_trigger=2, end_trigger=1), times)
    assert model.resolve_window(model.TraceParams(start_trigger=1, tend=5.0), times) == (1.0, 5.0)


def test_legacy_trigger_times_are_trace_edges():
    legacy = load_ptu(NO_PULSE)
    params = model.TraceParams(bintime=0.5, laser_pulse=False)
    with legacy.lock:
        times = model.trigger_times(legacy, params)
    assert times is not None and len(times) == 2


# --------------------------------------------------------------------- ROIs
ROI_A = {"x_min": 300.0, "x_max": 340.0, "y_min": 0.0, "y_max": 1e6}
ROI_B = {"x_min": 400.0, "x_max": 440.0, "y_min": 0.0, "y_max": 1e6}
EMPTY = {"x_min": 0.0, "x_max": 100.0, "y_min": 1e6, "y_max": 2e6}


def test_roi_mask_uses_time_overlap_and_count_range():
    signal = Signal(np.arange(5.0), np.array([1.0, 5.0, 9.0, 5.0, 1.0]),
                    meta={"bintime": 1.0})
    mask = model.roi_mask(signal, [{"x_min": 0.5, "x_max": 3.0, "y_min": 2.0, "y_max": 9.0}])
    assert mask.tolist() == [False, True, False, False, False]
    both = model.roi_mask(signal, [{"x_min": 0.0, "x_max": 5.0, "y_min": 9.0, "y_max": 10.0},
                                   {"x_min": 3.0, "x_max": 4.0, "y_min": 0.0, "y_max": 9.0}])
    assert both.tolist() == [False, False, True, True, False]


def test_roi_histograms_add_up(handle):
    params, states = model.TraceParams(bintime=1.0), model.StateParams()
    rois = [ROI_A, ROI_B]
    first = model.compute_histogram(handle, params, model.FitParams(source="roi", roi=0),
                                    states, rois)
    second = model.compute_histogram(handle, params, model.FitParams(source="roi", roi=1),
                                     states, rois)
    both = model.compute_histogram(handle, params, model.FitParams(source="rois"),
                                   states, rois)
    np.testing.assert_array_equal(first.y + second.y, both.y)
    assert second.label.startswith("ROI 2") and both.meta["roi_numbers"] == [1, 2]
    assert second.meta["acquisition_duration"] == pytest.approx(40.0)
    with pytest.raises(ValueError, match="No ROI"):
        model.compute_histogram(handle, params, model.FitParams(source="rois"), states, [])
    # The preview falls back to the whole trace instead of failing.
    preview = model.preview_histogram(handle, params, model.FitParams(source="each"),
                                      states, [])
    whole = model.compute_histogram(handle, params, model.FitParams(), states)
    assert preview.y.sum() == whole.y.sum()


def test_each_roi_is_fitted_and_empty_ones_are_skipped(handle, tmp_path):
    results, skipped = model.compute_roi_fits(
        handle, model.TraceParams(bintime=1.0), model.FitParams(source="each", n_exp=1),
        model.StateParams(), [ROI_A, EMPTY, ROI_B])
    assert [r.roi_index for r in results] == [0, 2]
    assert [i for i, _ in skipped] == [1]
    for r in results:
        summary = model.fit_summary(r.fit)
        assert summary["success"] and np.isfinite(summary["reduced_chi_squared"])
    payload = model.fits_payload(results)
    assert set(payload.metadata["results"]) == {"ROI 1", "ROI 3"}
    text = write_payload_csv(tmp_path / "rois.csv", payload).read_text(encoding="utf-8")
    header = next(line for line in text.splitlines() if not line.startswith("#"))
    assert "ROI 3 weighted residuals y (sigma)" in header


def test_roi_lifetime_trace_uses_trace_bin_time(handle):
    lt = model.LifetimeTraceParams(bintime=50.0, roi=0)
    signal = model.compute_lifetime_trace(handle, model.TraceParams(bintime=2.0),
                                          model.FitParams(), lt, rois=[ROI_A])
    assert signal.meta["bintime"] == 2.0 and signal.label.endswith("ROI 1")
    finite = signal.x[np.isfinite(signal.y)]
    assert finite.size and finite[0] >= ROI_A["x_min"] - 2.0 and finite[-1] < ROI_A["x_max"]
    with pytest.raises(ValueError, match="ROI 2"):
        model.compute_lifetime_trace(handle, model.TraceParams(bintime=2.0),
                                     model.FitParams(), model.LifetimeTraceParams(roi=1),
                                     rois=[ROI_A])


# -------------------------------------------------------------- fit options
def test_default_fit_options_change_nothing():
    options = model.FitOptions()
    assert options.is_default() and options.fit_kwargs(2, 1000.0) == {}


def test_fit_options_build_scaled_bounds_and_guesses():
    options = model.FitOptions()
    options.tau_min[1] = 2.0
    options.tau_guess[0] = 0.5
    options.amp_max[0] = 0.8
    options.amp_max[3] = 0.1          # beyond n_exp: ignored
    options.weighting, options.loss, options.f_scale = "poisson", "huber", 2.0
    kw = options.fit_kwargs(2, amplitude_scale=1000.0)
    lower, upper = kw["bounds"]
    np.testing.assert_allclose(lower, [0.0, 0.0, 1e-10, 2.0])
    np.testing.assert_allclose(upper, [800.0, np.inf, np.inf, np.inf])
    assert np.isnan(kw["p0"][[0, 1, 3]]).all() and kw["p0"][2] == 0.5
    assert kw["weighting"] == "poisson" and kw["loss"] == "huber" and kw["f_scale"] == 2.0
    options.weighting = "none"
    assert options.fit_kwargs(2, amplitude_scale=1000.0)["f_scale"] == 2000.0


def test_fit_options_validation():
    options = model.FitOptions()
    options.tau_min[0], options.tau_max[0] = 5.0, 5.0
    with pytest.raises(ValueError, match="τ1"):
        options.validate()
    options = model.FitOptions()
    options.amp_guess[1], options.amp_max[1] = 2.0, 1.0
    with pytest.raises(ValueError, match="A2"):
        options.validate()


def test_fit_respects_lifetime_bounds(handle):
    options = model.FitOptions()
    options.tau_min[0], options.tau_max[0] = 2.0, 2.5
    result = model.compute_fit(handle, model.TraceParams(bintime=1.0),
                               model.FitParams(n_exp=1), model.StateParams(), options=options)
    assert 2.0 <= result.fit["tau"][0] <= 2.5
    assert result.options is options
    payload = model.fit_payload(result)
    assert payload.metadata["parameters"]["fit_options"]["tau_min"][0] == 2.0
