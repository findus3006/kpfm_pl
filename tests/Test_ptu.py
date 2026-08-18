from pathlib import Path

import numpy as np
import pytest

from kpfm_pl.core import Signal
from kpfm_pl.ptu import PTU

DATA_DIR = Path(__file__).parents[1] / "examples" / "Test data"
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


def test_time_trace_window_uses_actual_bins_and_keeps_partial_final_bin():
    p = _synthetic_ptu(np.ones((3, 4), dtype=np.uint32), bintime=0.5,
                       acquisition_time=1.1)
    tt = p.time_trace(bintime=0.5, tstart=0.26, tend=1.1)
    np.testing.assert_allclose(tt.x, [0.0, 0.5, 1.0])
    assert tt.meta["matrix"].shape[0] == 3


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
    np.testing.assert_array_equal(tt.y, tt.meta["matrix"].sum(axis=1))


def test_legacy_trace_and_matrix_receive_the_same_transform():
    matrix = np.zeros((8, 3), dtype=np.uint32)
    matrix[:3] = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    p = _synthetic_ptu(matrix, bintime=1.0, acquisition_time=8.0)
    tt = p.time_trace(bintime=1.0, laser_pulse=False)
    np.testing.assert_array_equal(tt.y, tt.meta["matrix"].sum(axis=1))
    assert np.flatnonzero(tt.y).tolist() == [2, 3, 4]


def test_threshold_histograms_record_selected_duration():
    matrix = np.array([[4, 0], [1, 0], [0, 0], [3, 0]], dtype=np.uint32)
    p = _synthetic_ptu(matrix, bintime=0.25, acquisition_time=1.0)
    p.time_trace(bintime=0.25)
    states = p.threshold(off=0.5, on=0.75)
    assert states["on"].meta["time_bin_count"] == 2
    assert states["on"].meta["acquisition_duration"] == pytest.approx(0.5)
    assert states["off"].meta["time_bin_count"] == 2


def test_fit_lifetime_uses_histogram_duration_for_dark_offset(monkeypatch):
    captured = {}

    class DummyFit:
        def __init__(self, t, y, n_exp, offset, **kwargs):
            captured["offset"] = offset

        def to_dict(self):
            return {"success": True}

    monkeypatch.setattr("kpfm_pl.ptu.MultiExpFit", DummyFit)
    p = _synthetic_ptu(np.ones((2, 4), dtype=np.uint32), bintime=0.25,
                       acquisition_time=0.5)
    p.dark_counts = 40.0
    hist = p.lifetime_histogram(p._matrix, bintime=0.25)
    p.fit_lifetime(hist, n_exp=1, mode="manual", start=-1, end=10)
    assert captured["offset"] == pytest.approx(5.0)

    missing = p.lifetime_histogram(p._matrix)
    with pytest.raises(ValueError, match="acquisition duration"):
        p.fit_lifetime(missing, n_exp=1, mode="manual", start=-1, end=10)


def test_flid_preserves_raw_counts_and_odd_lifetime_groups(monkeypatch):
    class DummyFit:
        success = True

        def __init__(self, *args, **kwargs):
            pass

        def parameters(self):
            return {"tau": np.array([2.0])}

    monkeypatch.setattr("kpfm_pl.ptu.MultiExpFit", DummyFit)
    matrix = np.array([[4, 2, 1, 0, 0], [8, 4, 2, 0, 0]], dtype=np.uint32)
    p = _synthetic_ptu(matrix, bintime=1.0, acquisition_time=2.0)
    image = p.flid(bintime=1.0, bins=2, lifetime_bins=3,
                   intensity_bins=4, mode="manual", start=-1, end=10, sigma=0)
    np.testing.assert_allclose(image.meta["counts"], [7.0, 14.0])
    assert image.data.shape == (4, 3)


@pytest.mark.parametrize("name,value", [
    ("n_exp", 0), ("bins", 0), ("lifetime_bins", 0), ("intensity_bins", 1025),
])
def test_flid_validates_bin_parameters(name, value):
    p = _synthetic_ptu(np.ones((2, 5), dtype=np.uint32), bintime=1.0,
                       acquisition_time=2.0)
    with pytest.raises(ValueError):
        p.flid(**{name: value})


def test_laser_triggers_map_times_to_reference_samples():
    p = _synthetic_ptu(np.ones((5, 2), dtype=np.uint32), bintime=0.25,
                       acquisition_time=1.25)
    p.marker = {4: [2, 8]}
    p.rep_time = 0.1
    ref = Signal(np.array([0.0, 0.25, 0.5, 0.75, 1.0]), np.ones(5))
    triggers = p.laser_triggers(reference=ref)
    assert triggers == {
        "left": {"index": 1, "time": 0.25},
        "right": {"index": 3, "time": 0.75},
    }
    with pytest.warns(RuntimeWarning, match="outside"):
        assert p.laser_triggers(reference=Signal([0.5, 0.75], [1, 1])) is None


def test_invalid_file_raises(tmp_path):
    bad = tmp_path / "bad.ptu"
    bad.write_bytes(b"NOTPQTTTR" + b"\x00" * 64)
    with pytest.raises(ValueError, match="not a valid PTU file"):
        PTU(bad)


def _synthetic_ptu(matrix: np.ndarray, *, bintime: float,
                   acquisition_time: float) -> PTU:
    """Construct a decoded PTU instance for focused matrix-level tests."""
    p = PTU.__new__(PTU)
    p.filepath = Path("synthetic.ptu")
    p.dark_counts = 0.0
    p.acquisition_time = acquisition_time
    p.n_lt_bins = matrix.shape[1]
    p.lt_time = np.arange(matrix.shape[1], dtype=float) * 1e-9
    p._matrix = matrix
    p._bintime = bintime
    p._tt_cache = {}
    p.marker = {}
    return p
