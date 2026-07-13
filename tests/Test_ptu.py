from pathlib import Path

import numpy as np
import pytest

from kpfm_pl.core import Signal
from kpfm_pl.ptu import PTU

DATA_DIR = Path(__file__).parent / "Test data"
NO_PULSE = DATA_DIR / "ptu_test1_NoLaserPulse_(360s_400nW_532nm_000).ptu"
PULSE = DATA_DIR / "ptu_test2_LaserPulse(DR1_800nW_1200_ChargeAccu_000).ptu"

pytestmark = pytest.mark.skipif(not NO_PULSE.exists(), reason="PTU test file missing")


@pytest.fixture(scope="module")
def ptu():
    return PTU(NO_PULSE)


def test_header_tags_parsed(ptu):
    assert isinstance(ptu.tags, dict)
    assert ptu.num_records > 0
    assert ptu.acquisition_time > 0


def test_get_events_shape(ptu):
    events = ptu.get_events()
    assert events.ndim == 2
    assert events.shape[1] == 3
    assert events.shape[0] > 0


def test_time_trace_returns_signal(ptu):
    tt = ptu.time_trace(bintime=0.5)
    assert isinstance(tt, Signal)
    assert tt.units == {"x": "s", "y": "counts"}
    assert len(tt["x"]) == len(tt["y"])
    # total photons in the trace cannot exceed the recorded events
    assert tt["y"].sum() <= ptu.get_events().shape[0]


def test_time_trace_window_is_subset(ptu):
    full = ptu.time_trace(bintime=0.5)
    windowed = ptu.time_trace(bintime=0.5, tstart=0.0, tend=10.0)
    assert len(windowed["x"]) < len(full["x"])
    assert windowed["x"][0] >= 0.0


def test_lifetime_histogram(ptu):
    hist = ptu.lifetime_histogram()
    assert isinstance(hist, Signal)
    assert hist["y"].sum() > 0
    assert len(hist["x"]) == ptu.n_lt_bins


def test_threshold_splits_on_off(ptu):
    ptu.time_trace(bintime=0.5)
    states = ptu.threshold(off=0.2, on=0.8)
    assert set(states) == {"on", "off"}
    assert isinstance(states["on"], Signal)
    assert isinstance(states["off"], Signal)


def test_fit_lifetime_returns_quality(ptu):
    hist = ptu.lifetime_histogram()
    fit = ptu.fit_lifetime(hist, n_exp=1)
    assert "tau" in fit
    assert "t_fit" in fit and len(fit["t_fit"]) > 0


def test_laser_triggers_none_without_pulse(ptu):
    # this file has no laser-pulse edges and no hardware markers
    ptu.time_trace(bintime=0.5, laser_pulse=True)
    assert ptu.laser_triggers() is None


def test_laser_pulse_false_records_edge_triggers():
    p = PTU(NO_PULSE)
    tt = p.time_trace(bintime=0.5, laser_pulse=False)
    trig = tt["meta"].get("triggers")
    assert trig is not None
    assert trig["left"]["index"] < trig["right"]["index"]
    # laser_triggers falls back to these trace edges
    got = p.laser_triggers()
    assert got == trig


def test_invalid_file_raises(tmp_path):
    bad = tmp_path / "bad.ptu"
    bad.write_bytes(b"NOTPQTTTR" + b"\x00" * 64)
    with pytest.raises(ValueError, match="not a valid PTU file"):
        PTU(bad)
