"""Qt-free tests of the GUI's PTU workflow model."""

from pathlib import Path

import numpy as np
import pytest

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
    assert len(payload.signals) == 2
    write_payload_csv(tmp_path / "fit.csv", payload)
