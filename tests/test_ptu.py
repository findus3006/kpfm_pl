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
    # the two trace edges, in the shared point format
    assert len(trig) == 2
    assert trig[0]["index"] < trig[1]["index"]
    for point in trig:
        assert point["time"] == tt.x[point["index"]]
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
            return {"A": np.ones(1), "tau": np.array([2.0])}

    monkeypatch.setattr("kpfm_pl.ptu.MultiExpFit", DummyFit)
    matrix = np.array([[4, 2, 1, 0, 0], [8, 4, 2, 0, 0]], dtype=np.uint32)
    p = _synthetic_ptu(matrix, bintime=1.0, acquisition_time=2.0)
    image = p.flid(bintime=1.0, bins=2, lifetime_bins=3,
                   intensity_bins=4, mode="manual", start=-1, end=10, sigma=0)
    np.testing.assert_allclose(image.meta["counts"], [7.0, 14.0])
    assert image.data.shape == (4, 3)


def test_lifetime_trace_preserves_time_bins_and_marks_failed_fits(monkeypatch):
    class DummyFit:
        calls = 0

        def __init__(self, *args, **kwargs):
            self.call = DummyFit.calls
            DummyFit.calls += 1
            self.success = self.call == 0

        def parameters(self):
            return {"A": np.ones(2), "tau": np.array([2.0, 4.0])}

    monkeypatch.setattr("kpfm_pl.ptu.MultiExpFit", DummyFit)
    matrix = np.array([
        [4, 2, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [8, 4, 2, 0, 0],
    ], dtype=np.uint32)
    p = _synthetic_ptu(matrix, bintime=1.0, acquisition_time=3.0)

    trace = p.lifetime_trace(
        bintime=1.0, n_exp=2, bins=2,
        mode="manual", start=-1, end=10,
    )

    np.testing.assert_allclose(trace.x, [0.0, 1.0, 2.0])
    assert len(trace.x) == len(trace.y) == matrix.shape[0]
    # intensity-weighted mean: (1*2**2 + 1*4**2) / (1*2 + 1*4)
    assert trace.y[0] == pytest.approx(20.0 / 6.0)
    assert np.isnan(trace.y[1])  # empty photon bin
    assert np.isnan(trace.y[2])  # unsuccessful fit
    assert trace.meta["taus"].shape == trace.meta["amplitudes"].shape == (3, 2)
    assert trace.units == {"x": "s", "y": "ns"}


@pytest.mark.parametrize("amps,expected", [
    ([3.0, 1.0], (3 * 1 + 1 * 16) / (3 * 1 + 1 * 4)),   # weighted towards tau=1
    ([1.0, 0.0], 1.0),                                  # only the first component
    ([0.0, 0.0], np.nan),                               # no weight at all
])
def test_lifetime_trace_uses_intensity_weighted_mean(monkeypatch, amps, expected):
    class DummyFit:
        success = True

        def __init__(self, *args, **kwargs):
            pass

        def parameters(self):
            return {"A": np.array(amps), "tau": np.array([1.0, 4.0])}

    monkeypatch.setattr("kpfm_pl.ptu.MultiExpFit", DummyFit)
    p = _synthetic_ptu(np.array([[4, 2, 1, 0]], dtype=np.uint32), bintime=1.0,
                       acquisition_time=1.0)
    trace = p.lifetime_trace(bintime=1.0, n_exp=2, mode="manual", start=-1, end=10)
    np.testing.assert_allclose(trace.y, [expected])


def test_lifetime_trace_keeps_intensity_roi_bins_as_nan(monkeypatch):
    class DummyFit:
        success = True

        def __init__(self, *args, **kwargs):
            pass

        def parameters(self):
            return {"A": np.ones(1), "tau": np.array([5.0])}

    monkeypatch.setattr("kpfm_pl.ptu.MultiExpFit", DummyFit)
    matrix = np.array([[4, 2, 1, 0], [8, 4, 2, 0]], dtype=np.uint32)
    p = _synthetic_ptu(matrix, bintime=1.0, acquisition_time=2.0)
    trace = p.lifetime_trace(
        bintime=1.0, y_min=10.0, mode="manual", start=-1, end=10,
    )

    np.testing.assert_allclose(trace.x, [0.0, 1.0])
    assert np.isnan(trace.y[0])
    assert trace.y[1] == pytest.approx(5.0)


def test_lifetime_trace_inherits_legacy_time_bin_transform(monkeypatch):
    class DummyFit:
        success = True

        def __init__(self, *args, **kwargs):
            pass

        def parameters(self):
            return {"A": np.ones(1), "tau": np.array([3.0])}

    monkeypatch.setattr("kpfm_pl.ptu.MultiExpFit", DummyFit)
    matrix = np.zeros((8, 4), dtype=np.uint32)
    matrix[:3] = [[4, 2, 1, 0], [8, 4, 2, 0], [12, 6, 3, 0]]
    p = _synthetic_ptu(matrix, bintime=1.0, acquisition_time=8.0)

    time_trace = p.time_trace(bintime=1.0, laser_pulse=False)
    lifetime_trace = p.lifetime_trace(
        bintime=1.0, mode="manual", start=-1, end=10,
    )

    assert len(lifetime_trace.x) == len(lifetime_trace.y) == len(time_trace.x)
    np.testing.assert_array_equal(lifetime_trace.x, time_trace.x)
    np.testing.assert_array_equal(
        np.flatnonzero(np.isfinite(lifetime_trace.y)),
        np.flatnonzero(time_trace.y),
    )
    assert lifetime_trace.meta["laser_pulse"] is False


def test_lifetime_trace_can_override_inherited_legacy_mode(monkeypatch):
    class DummyFit:
        success = True

        def __init__(self, *args, **kwargs):
            pass

        def parameters(self):
            return {"A": np.ones(1), "tau": np.array([3.0])}

    monkeypatch.setattr("kpfm_pl.ptu.MultiExpFit", DummyFit)
    matrix = np.zeros((8, 4), dtype=np.uint32)
    matrix[:3] = [[4, 2, 1, 0], [8, 4, 2, 0], [12, 6, 3, 0]]
    p = _synthetic_ptu(matrix, bintime=1.0, acquisition_time=8.0)
    p.time_trace(bintime=1.0, laser_pulse=False)

    lifetime_trace = p.lifetime_trace(
        bintime=1.0, laser_pulse=True,
        mode="manual", start=-1, end=10,
    )

    np.testing.assert_array_equal(
        np.flatnonzero(np.isfinite(lifetime_trace.y)), [0, 1, 2]
    )
    assert lifetime_trace.meta["laser_pulse"] is True


def _flid_ptu(monkeypatch, taus):
    """Synthetic PTU whose per-bin fits return ``taus`` in order."""
    fitted = iter(taus)

    class DummyFit:
        success = True

        def __init__(self, *args, **kwargs):
            self.tau = next(fitted)

        def parameters(self):
            return {"A": np.ones(1), "tau": np.array([self.tau])}

    monkeypatch.setattr("kpfm_pl.ptu.MultiExpFit", DummyFit)
    matrix = np.array([[4, 2, 1, 0], [8, 4, 2, 0], [12, 6, 3, 0]], dtype=np.uint32)
    return _synthetic_ptu(matrix, bintime=1.0, acquisition_time=3.0)


def test_flid_explicit_ranges_set_histogram_bins(monkeypatch):
    p = _flid_ptu(monkeypatch, [2.0, 4.0, 6.0])
    image = p.flid(bintime=1.0, lifetime_bins=10, intensity_bins=5,
                   lifetime_range=(0, 10), intensity_range=(0, 25),
                   mode="manual", start=-1, end=10, sigma=0)
    assert image.meta["lifetime_range"] == (0.0, 10.0)
    assert image.meta["intensity_range"] == (0.0, 25.0)
    np.testing.assert_allclose(image.x, np.arange(0.5, 10, 1.0))
    np.testing.assert_allclose(image.y, [2.5, 7.5, 12.5, 17.5, 22.5])
    # counts 7, 14, 21 with taus 2, 4, 6 land in their own (count, tau) cells
    assert image.data[1, 2] == image.data[2, 4] == image.data[4, 6] == 1
    assert image.data.sum() == 3


def test_flid_reports_progress_per_fitted_bin(monkeypatch):
    p = _flid_ptu(monkeypatch, [2.0, 4.0, 6.0])
    calls = []
    p.flid(bintime=1.0, mode="manual", start=-1, end=10, sigma=0,
           progress=lambda done, total: calls.append((done, total)))
    assert calls == [(0, 3), (1, 3), (2, 3), (3, 3)]


def test_lifetime_trace_progress_callback_can_abort(monkeypatch):
    p = _flid_ptu(monkeypatch, [2.0, 4.0, 6.0])

    class Cancelled(Exception):
        pass

    def cancel_after_first(done, total):
        if done == 1:
            raise Cancelled

    with pytest.raises(Cancelled):
        p.lifetime_trace(bintime=1.0, mode="manual", start=-1, end=10,
                         progress=cancel_after_first)


def test_flid_explicit_ranges_drop_values_outside(monkeypatch):
    p = _flid_ptu(monkeypatch, [2.0, 4.0, 6.0])
    image = p.flid(bintime=1.0, lifetime_range=(0, 5), intensity_range=(0, 25),
                   mode="manual", start=-1, end=10, sigma=0)
    assert image.data.sum() == 2


def test_flid_auto_range_pads_for_smoothing(monkeypatch):
    p = _flid_ptu(monkeypatch, [2.0, 4.0, 6.0])
    image = p.flid(bintime=1.0, lifetime_bins=20, intensity_bins=20,
                   mode="manual", start=-1, end=10, sigma=1.0)
    # data spans tau 2..6 on 20 - 2*3 bins, plus 3 empty bins each side
    width = 4.0 / 13
    assert image.meta["lifetime_range"] == pytest.approx((2 - 3 * width, 6 + 4 * width))
    assert image.meta["intensity_range"][0] == pytest.approx(7 - 3 * 14 / 13)
    tau_counts = np.histogram([2.0, 4.0, 6.0], bins=20, range=image.meta["lifetime_range"])[0]
    assert np.flatnonzero(tau_counts).tolist() == [3, 9, 16]
    # the smoothing tails fit on the map: edges are only ~1 % of the peak
    assert image.data.sum() == pytest.approx(3.0, rel=1e-2)
    peak = image.data.max()
    assert image.data[:, 0].max() < 0.02 * peak and image.data[:, -1].max() < 0.02 * peak


def test_flid_auto_range_does_not_pad_below_zero(monkeypatch):
    p = _flid_ptu(monkeypatch, [0.1, 4.0, 6.0])
    # 15 padding bins of width span/10 reach far below zero on both axes
    image = p.flid(bintime=1.0, lifetime_bins=40, intensity_bins=40,
                   mode="manual", start=-1, end=10, sigma=5.0)
    assert image.meta["lifetime_range"][0] == 0.0
    assert image.meta["intensity_range"][0] == 0.0


def test_flid_without_smoothing_spans_data_exactly(monkeypatch):
    p = _flid_ptu(monkeypatch, [2.0, 4.0, 6.0])
    image = p.flid(bintime=1.0, mode="manual", start=-1, end=10, sigma=0)
    assert image.meta["lifetime_range"] == (2.0, 6.0)
    assert image.meta["intensity_range"] == (7.0, 21.0)


def _two_component_flid(monkeypatch, amps, taus, **kwargs):
    class DummyFit:
        success = True

        def __init__(self, *args, **kwargs):
            pass

        def parameters(self):
            return {"A": np.array(amps, dtype=float), "tau": np.array(taus, dtype=float)}

    monkeypatch.setattr("kpfm_pl.ptu.MultiExpFit", DummyFit)
    p = _synthetic_ptu(np.array([[4, 2, 1, 0]], dtype=np.uint32), bintime=1.0,
                       acquisition_time=1.0)
    return p.flid(bintime=1.0, n_exp=2, lifetime_bins=10, lifetime_range=(0, 10),
                  intensity_range=(0, 10), mode="manual", start=-1, end=10,
                  sigma=0, **kwargs)


def test_flid_uses_intensity_weighted_mean_lifetime(monkeypatch):
    image = _two_component_flid(monkeypatch, [1.0, 1.0], [1.0, 4.0])
    # (1*1 + 1*16) / (1*1 + 1*4) = 3.4 ns, not the arithmetic mean 2.5 ns
    assert np.flatnonzero(image.data.sum(axis=0)).tolist() == [3]
    np.testing.assert_allclose(image.meta["amplitudes"], [[1.0, 1.0]])
    np.testing.assert_allclose(image.meta["taus"], [[1.0, 4.0]])


def test_flid_isolated_maps_are_not_weighted(monkeypatch):
    images = _two_component_flid(monkeypatch, [1.0, 1.0], [1.0, 4.0], isolated=True)
    assert [np.flatnonzero(im.data.sum(axis=0)).tolist() for im in images] == [[1], [4]]


def test_flid_drops_bins_with_zero_weights(monkeypatch):
    image = _two_component_flid(monkeypatch, [0.0, 0.0], [1.0, 4.0])
    assert image.data.sum() == 0


@pytest.mark.parametrize("name,value", [
    ("lifetime_range", (5, 1)), ("lifetime_range", (0, np.inf)),
    ("intensity_range", (1,)), ("intensity_range", (3, 3)),
])
def test_flid_validates_ranges(name, value):
    p = _synthetic_ptu(np.ones((2, 5), dtype=np.uint32), bintime=1.0,
                       acquisition_time=2.0)
    with pytest.raises(ValueError, match=name):
        p.flid(**{name: value})


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
    assert triggers == [{"index": 1, "time": 0.25}, {"index": 3, "time": 0.75}]
    with pytest.warns(RuntimeWarning, match="outside"):
        assert p.laser_triggers(reference=Signal([2.0, 2.25], [1, 1])) is None


def test_laser_triggers_return_every_marker():
    p = _synthetic_ptu(np.ones((5, 2), dtype=np.uint32), bintime=0.25,
                       acquisition_time=1.25)
    p.marker = {4: [2, 5, 8]}
    p.rep_time = 0.1
    ref = Signal(np.array([0.0, 0.25, 0.5, 0.75, 1.0]), np.ones(5))
    triggers = p.laser_triggers(reference=ref)
    assert [point["index"] for point in triggers] == [1, 2, 3]
    for point in triggers:
        assert point["time"] == ref.x[point["index"]]


def test_invalid_file_raises(tmp_path):
    bad = tmp_path / "bad.ptu"
    bad.write_bytes(b"NOTPQTTTR" + b"\x00" * 64)
    with pytest.raises(ValueError, match="not a valid PTU file"):
        PTU(bad)


def test_laser_triggers_single_marker_is_a_one_point_list():
    p = _synthetic_ptu(np.ones((5, 2), dtype=np.uint32), bintime=0.25,
                       acquisition_time=1.25)
    p.marker = {4: [2]}
    p.rep_time = 0.1
    ref = Signal(np.array([0.0, 0.25, 0.5, 0.75, 1.0]), np.ones(5))
    assert p.laser_triggers(reference=ref) == [{"index": 1, "time": 0.25}]


def test_time_trace_y_is_float():
    p = _synthetic_ptu(np.ones((3, 4), dtype=np.uint32), bintime=0.5,
                       acquisition_time=1.5)
    tt = p.time_trace(bintime=0.5)
    assert np.issubdtype(tt["y"].dtype, np.floating)
    # subtraction must go negative instead of wrapping like uint32
    assert np.min(tt["y"] - (np.max(tt["y"]) + 1.0)) < 0


def test_build_matrix_grows_for_photons_past_declared_duration():
    p = _synthetic_ptu(np.zeros((2, 4), dtype=np.uint32), bintime=0.5,
                       acquisition_time=1.0)
    p.rep_time = 0.1
    # the last photon (sync 12 -> time bin 2) indexes past the two bins
    # implied by the header acquisition time
    p.events = np.array([[2.0, 1.0, 0.2], [12.0, 2.0, 1.2]])
    p._build_matrix(0.5)
    assert p._matrix.shape == (3, 4)
    assert p._matrix.sum() == 2


def _header_tag(name: str, typ: int, value: bytes = b"\x00" * 8) -> bytes:
    import struct

    return name.encode().ljust(32, b"\x00") + struct.pack("<ii", -1, typ) + value


def test_continuous_wave_file_constructs_and_rejects_binning(tmp_path):
    import struct

    from kpfm_pl.ptu import _TAG_TYPES

    data = (
        b"PQTTTR\x00\x00" + b"1.0.00\x00\x00"
        + _header_tag("TTResult_NumberOfRecords", _TAG_TYPES["Int8"],
                      struct.pack("<q", 0))
        + _header_tag("MeasDesc_GlobalResolution", _TAG_TYPES["Float8"],
                      struct.pack("<d", np.inf))
        + _header_tag("MeasDesc_Resolution", _TAG_TYPES["Float8"],
                      struct.pack("<d", 4e-12))
        + _header_tag("MeasDesc_AcquisitionTime", _TAG_TYPES["Int8"],
                      struct.pack("<q", 360000))
        + _header_tag("Header_End", _TAG_TYPES["Empty8"])
    )
    path = tmp_path / "cw.ptu"
    path.write_bytes(data)

    p = PTU(path)
    # fully constructed object with safe defaults
    assert p.is_cw is True
    assert p.acquisition_time == pytest.approx(360.0)
    assert p.n_lt_bins == 0
    assert p.get_events().shape[0] == 0
    # time-binned / lifetime analysis raises a clear error instead of an
    # AttributeError from a half-constructed object
    for call in (p.time_trace, p.lifetime_histogram, p.lifetime_trace, p.flid, p.threshold,
                 p.laser_triggers, p.get_markers):
        with pytest.raises(ValueError, match="continuous-wave"):
            call()


def _synthetic_ptu(matrix: np.ndarray, *, bintime: float,
                   acquisition_time: float) -> PTU:
    """Construct a decoded PTU instance for focused matrix-level tests."""
    p = PTU.__new__(PTU)
    p.filepath = Path("synthetic.ptu")
    p.dark_counts = 0.0
    p.is_cw = False
    p.acquisition_time = acquisition_time
    p.n_lt_bins = matrix.shape[1]
    p.lt_time = np.arange(matrix.shape[1], dtype=float) * 1e-9
    p._matrix = matrix
    p._bintime = bintime
    p._tt_cache = {}
    p.marker = {}
    return p


def test_fit_lifetime_reports_poisson_quality_and_weighting(ptu):
    hist = ptu.lifetime_histogram()
    plain = ptu.fit_lifetime(hist, n_exp=1)
    q = plain["quality"]
    assert {"reduced_chi_squared", "chi_squared", "dof", "weighted_residuals"} <= set(q)
    assert len(q["weighted_residuals"]) == len(plain["t_fit"])
    assert q["dof"] == len(plain["t_fit"]) - 2
    assert plain["weighting"] == "none"
    # chi-squared is computed on raw counts, so normalising does not change it much
    norm = ptu.fit_lifetime(hist, n_exp=1, normalize=True)
    assert norm["quality"]["reduced_chi_squared"] == pytest.approx(q["reduced_chi_squared"],
                                                                   rel=0.05)
    weighted = ptu.fit_lifetime(hist, n_exp=1, weighting="poisson")
    assert weighted["success"] and weighted["weighting"] == "poisson"
    # Poisson weighting minimises exactly this chi-squared, so it cannot be worse.
    assert weighted["quality"]["chi_squared"] <= q["chi_squared"] * (1 + 1e-6)


def test_weighting_is_validated(ptu):
    hist = ptu.lifetime_histogram()
    with pytest.raises(ValueError, match="weighting"):
        ptu.fit_lifetime(hist, n_exp=1, weighting="sqrt")
    with pytest.raises(ValueError, match="either"):
        ptu.fit_lifetime(hist, n_exp=1, weighting="poisson", weights=np.ones(3))
    with pytest.raises(ValueError, match="weighting"):
        ptu.lifetime_trace(bintime=50.0, weighting="sqrt")


def test_lifetime_trace_passes_poisson_weights_per_bin(monkeypatch):
    seen = []

    class DummyFit:
        def __init__(self, t, y, n_exp, offset, **kwargs):
            seen.append(kwargs.get("weights"))
            self.success = True

        def parameters(self):
            return {"A": np.ones(1), "tau": np.array([2.0])}

    monkeypatch.setattr("kpfm_pl.ptu.MultiExpFit", DummyFit)
    matrix = np.array([[16, 4, 1, 0, 0], [64, 16, 4, 1, 0]], dtype=np.uint32)
    p = _synthetic_ptu(matrix, bintime=1.0, acquisition_time=2.0)
    p.lifetime_trace(bintime=1.0, mode="manual", start=-1, end=10, weighting="poisson")
    assert len(seen) == 2 and all(w is not None for w in seen)
    # sigma units on normalised data: weight = peak / sqrt(counts), sqrt(peak) at the peak
    assert seen[0][0] == pytest.approx(16 / 4) and seen[1][0] == pytest.approx(64 / 8)


# ------------------------------------------------------------ lifetime shift
WRAPPED = DATA_DIR / "test3_DR1_360s_10Hz_4.67W_Scan1_001.ptu"


def _wrapped_ptu(peak_ns: float = 80.0) -> PTU:
    """A real file whose raw lifetimes are rotated so the peak sits at ``peak_ns``."""
    p = PTU(WRAPPED)
    wrap = int(np.ceil(p.rep_time / p.resolution - 1e-6))
    peak = int(np.argmax(np.bincount(p._raw_dtime.astype(int), minlength=wrap)))
    k = int(round(peak_ns * 1e-9 / p.resolution)) - peak
    p._raw_dtime = (p._raw_dtime + k) % wrap
    p.set_lifetime_shift(None)
    return p


@pytest.mark.skipif(not WRAPPED.exists(), reason="PTU test file missing")
def test_auto_lifetime_shift_repairs_a_wrapped_decay():
    reference = PTU(WRAPPED)
    expected = reference.fit_lifetime(reference.lifetime_histogram(), n_exp=2)

    p = _wrapped_ptu(80.0)
    hist = p.lifetime_histogram()
    assert hist.x[np.argmax(hist.y)] * 1e9 == pytest.approx(80.0, abs=0.2)
    shift = p.set_lifetime_shift("auto")
    assert p.lifetime_shift == "auto" and shift == pytest.approx(75.0, abs=0.5)
    hist = p.lifetime_histogram()
    assert hist.x[np.argmax(hist.y)] * 1e9 == pytest.approx(5.0, abs=0.3)
    assert hist.meta["lifetime_shift_ns"] == shift
    # photons are only moved, never lost or added
    assert hist.y.sum() == reference.lifetime_histogram().y.sum()

    fit = p.fit_lifetime(hist, n_exp=2)
    assert fit["lifetime_shift_ns"] == shift
    np.testing.assert_allclose(np.sort(fit["tau"]), np.sort(expected["tau"]), rtol=0.05)
    # The partial last bin of the period (1562.5 bins) lies inside the decay
    # now and is left out of the fit.
    assert p.lifetime_seam_bin is not None and fit["seam_ns"] is not None
    assert fit["seam_ns"] not in fit["t_fit"]
    assert fit["t_fit"][-1] > fit["seam_ns"]


@pytest.mark.skipif(not WRAPPED.exists(), reason="PTU test file missing")
def test_manual_lifetime_shift_is_reversible_and_resets_caches():
    p = PTU(WRAPPED)
    raw = p.events[:, 1].copy()
    before = p.time_trace(bintime=1.0)
    shift = p.set_lifetime_shift(10.0)
    assert shift == pytest.approx(round(10e-9 / p.resolution) * p.resolution * 1e9)
    assert not np.array_equal(p.events[:, 1], raw)
    # the cached matrix was rebuilt: same trace, rotated lifetimes
    after = p.time_trace(bintime=1.0)
    np.testing.assert_array_equal(before.y, after.y)
    assert not np.array_equal(before.meta["matrix"], after.meta["matrix"])
    assert p.lifetime_trace(bintime=50.0).meta["lifetime_shift_ns"] == shift
    assert p.set_lifetime_shift(None) == 0.0 and p.lifetime_seam_bin is None
    np.testing.assert_array_equal(p.events[:, 1], raw)
    assert "lifetime_shift_ns" not in p.lifetime_histogram().meta


@pytest.mark.skipif(not WRAPPED.exists(), reason="PTU test file missing")
def test_lifetime_shift_validation():
    p = PTU(WRAPPED)
    with pytest.raises(ValueError, match="number, None or 'auto'"):
        p.set_lifetime_shift("left")
    p.n_lt_bins = 1000      # a window shorter than the period is not periodic
    with pytest.raises(ValueError, match="shorter than one laser period"):
        p.set_lifetime_shift(5.0)
    q = PTU(WRAPPED, lifetime_shift="auto")
    assert q.lifetime_shift_ns > 0
