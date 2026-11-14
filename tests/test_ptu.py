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
    with pytest.raises(ValueError, match="loss"):
        ptu.fit_lifetime(hist, n_exp=1, weighting="mle", loss="soft_l1")


def test_fit_lifetime_reports_bic_warnings_and_period_correction(ptu):
    hist = ptu.lifetime_histogram()
    one = ptu.fit_lifetime(hist, n_exp=1, weighting="poisson")
    two = ptu.fit_lifetime(hist, n_exp=2, weighting="poisson")
    for fit, n in ((one, 1), (two, 2)):
        q = fit["quality"]
        assert q["bic_statistic"] == "chi_squared" and q["n_params"] == 2 * n
        assert q["bic"] == pytest.approx(q["chi_squared"] + 2 * n * np.log(q["n_points"]))
        assert isinstance(fit["warnings"], list)
        # per-pulse amplitudes: A = A_fit * (1 - exp(-T/tau))
        period_ns = ptu.rep_time * 1e9
        np.testing.assert_allclose(fit["A"],
                                   fit["A_fit"] * (1 - np.exp(-period_ns / fit["tau"])))
        assert fit["period"] == pytest.approx(period_ns)
    # the second component is needed (see the chi-squared note in Changes.md)
    assert two["quality"]["bic"] < one["quality"]["bic"]
    strict = ptu.fit_lifetime(hist, n_exp=2, weighting="poisson",
                              diagnostics={"min_fraction": 0.99})
    assert any("collapsed" in w for w in strict["warnings"])


def test_fit_lifetime_mean_lifetime_and_decay_correction_switch(ptu):
    hist = ptu.lifetime_histogram()
    fit = ptu.fit_lifetime(hist, n_exp=2, weighting="poisson")
    assert fit["mean_lifetime"] == "intensity" and fit["decay_correction"]
    assert fit["tau_mean"] == fit["tau_int"] and fit["tau_mean_err"] == fit["tau_int_err"]
    np.testing.assert_allclose(fit["f_amp"], fit["A"] / np.sum(fit["A"]))
    amp = ptu.fit_lifetime(hist, n_exp=2, weighting="poisson", mean_lifetime="amplitude")
    assert amp["tau_mean"] == pytest.approx(amp["tau_amp"])
    assert amp["tau_amp"] < amp["tau_int"]      # the intensity weighting favours long tau
    raw = ptu.fit_lifetime(hist, n_exp=2, weighting="poisson", decay_correction=False)
    assert raw["period"] is None and not raw["decay_correction"]
    np.testing.assert_allclose(raw["A"], raw["A_fit"])
    np.testing.assert_allclose(raw["A_fit"], fit["A_fit"])
    assert raw["tau_int"] == pytest.approx(fit["tau_int_fit"])
    with pytest.raises(ValueError, match="mean_lifetime"):
        ptu.fit_lifetime(hist, n_exp=1, mean_lifetime="median")


def test_lifetime_trace_dark_offset_is_in_fit_units(monkeypatch):
    seen = []

    class DummyFit:
        def __init__(self, t, y, n_exp, offset, **kwargs):
            seen.append((np.asarray(y, dtype=float), offset))
            self.success = True

        def parameters(self):
            return {"A": np.ones(1), "tau": np.array([2.0]), "A_fit": np.ones(1)}

    monkeypatch.setattr("kpfm_pl.ptu.MultiExpFit", DummyFit)
    matrix = np.array([[16, 4, 2, 1, 0]], dtype=np.uint32)
    p = _synthetic_ptu(matrix, bintime=1.0, acquisition_time=1.0)
    p.dark_counts = 10.0                       # 10 counts/s over 5 lifetime bins: 2 per bin
    # The window starts after the peak, so its maximum (4) is below the row's (16).
    p.lifetime_trace(bintime=1.0, mode="manual", start=0.5, end=10, min_counts=None)
    y, offset = seen[0]
    np.testing.assert_allclose(y, [1.0, 0.5, 0.25, 0.0])        # counts / 4
    assert offset == pytest.approx(2.0 / 4.0)                   # same units as y


def test_lifetime_trace_mean_lifetime_and_decay_correction(monkeypatch):
    seen = []

    class DummyFit:
        def __init__(self, t, y, n_exp, offset, **kwargs):
            seen.append(kwargs.get("period"))
            self.success = True

        def parameters(self):
            return {"A": np.array([0.5, 0.5]), "tau": np.array([1.0, 3.0]),
                    "A_fit": np.array([0.5, 0.5])}

    monkeypatch.setattr("kpfm_pl.ptu.MultiExpFit", DummyFit)
    matrix = np.array([[16, 4, 1, 0, 0], [64, 16, 4, 1, 0]], dtype=np.uint32)
    p = _synthetic_ptu(matrix, bintime=1.0, acquisition_time=2.0)
    p.rep_time = 50e-9
    kw = dict(bintime=1.0, n_exp=2, mode="manual", start=-1, end=10)
    intensity = p.lifetime_trace(**kw)
    np.testing.assert_allclose(intensity.y, (0.5 * 1 + 0.5 * 9) / (0.5 + 1.5))
    amplitude = p.lifetime_trace(mean_lifetime="amplitude", decay_correction=False, **kw)
    np.testing.assert_allclose(amplitude.y, 2.0)
    assert amplitude.meta["mean_lifetime"] == "amplitude"
    assert not amplitude.meta["decay_correction"]
    assert seen[:2] == [pytest.approx(50.0)] * 2 and seen[2:] == [None, None]
    with pytest.raises(ValueError, match="mean_lifetime"):
        p.lifetime_trace(mean_lifetime="median", **kw)


def _irf_ptu(rows=4, n=500, dt=0.1, tau=3.0, shift=0.2):
    """Synthetic file with one exact laser period of ``n`` bins and its IRF."""
    from kpfm_pl import IRFReconvolutionFit

    t = np.arange(n) * dt
    irf = np.exp(-0.5 * ((t - 5.0) / 0.15) ** 2) * 1e4 + 5
    model = IRFReconvolutionFit(irf, np.ones(n), 1, dt, shift=shift, background=50.0)
    decay = model.full_model(np.array([4e4, tau]))       # high counts: rounding is negligible
    matrix = np.round(np.tile(decay, (rows, 1))).astype(np.uint32)
    p = _synthetic_ptu(matrix, bintime=1.0, acquisition_time=float(rows))
    p.resolution, p.rep_time = dt * 1e-9, n * dt * 1e-9
    p.lt_time = t * 1e-9
    return p, irf


def test_fit_lifetime_with_irf_reconvolution():
    p, irf = _irf_ptu()
    hist = p.lifetime_histogram(p._matrix, bintime=1.0)
    fit = p.fit_lifetime(hist, n_exp=1, irf=irf, weighting="poisson", fit_background=True)
    assert fit["success"] and fit["reconvolution"]
    assert fit["tau"][0] == pytest.approx(3.0, rel=0.01)
    assert fit["shift_ns"] == pytest.approx(0.2, abs=0.01)
    assert fit["background"] == pytest.approx(4 * 50.0, rel=0.01)   # 4 rows of 50
    assert fit["quality"]["n_params"] == 4                         # A, tau, shift, B
    assert fit["t_fit"].size == 500                                # whole period
    assert fit["period"] == pytest.approx(50.0) and fit["irf_t_ns"].size == 500
    manual = p.fit_lifetime(hist, n_exp=1, irf=irf, mode="manual", start=-2.0, end=20.0)
    assert manual["t_fit"].size < 500 and manual["t_fit"][0] >= -2.0
    with pytest.raises(ValueError, match="normalize"):
        p.fit_lifetime(hist, n_exp=1, irf=irf, normalize=True)
    with pytest.raises(ValueError, match="lifetime bins"):
        p.fit_lifetime(hist, n_exp=1, irf=irf[:-1])


def test_irf_needs_a_whole_period():
    p, irf = _irf_ptu()
    p.rep_time = 80e-9                        # the 50 ns window is shorter than 80 ns
    hist = p.lifetime_histogram(p._matrix, bintime=1.0)
    with pytest.raises(ValueError, match="shorter than one laser period"):
        p.fit_lifetime(hist, n_exp=1, irf=irf)


@pytest.mark.skipif(not PULSE.exists(), reason="PTU test file missing")
def test_irf_file_must_match_resolution_and_period(ptu):
    other = PTU(PULSE)
    assert not np.isclose(other.rep_time, ptu.rep_time, rtol=1e-4, atol=0)
    with pytest.raises(ValueError, match="laser period"):
        other.irf_histogram(ptu)
    assert other.irf_histogram(other).sum() == other.events.shape[0]


def test_lifetime_trace_with_irf_uses_fixed_shift():
    p, irf = _irf_ptu(rows=3, tau=2.0)
    trace = p.lifetime_trace(bintime=1.0, irf=irf, irf_shift=0.2, background_rate=50.0,
                             min_counts=None)
    assert trace.meta["reconvolution"] and trace.meta["irf_shift_ns"] == 0.2
    np.testing.assert_allclose(trace.y, 2.0, rtol=0.01)
    # per-bin amplitudes are normalised to the bin's peak, as for the tail fits
    assert np.all(trace.meta["amplitudes"] > 0)


def test_fit_lifetime_mle(ptu):
    hist = ptu.lifetime_histogram()
    neyman = ptu.fit_lifetime(hist, n_exp=2, weighting="poisson")
    mle = ptu.fit_lifetime(hist, n_exp=2, weighting="mle")
    assert mle["success"] and mle["statistic"] == "poisson_mle"
    assert mle["quality"]["bic_statistic"] == "deviance"
    assert mle["quality"]["deviance"] <= neyman["quality"]["deviance"] * (1 + 1e-6)
    # The tail holds ~10 counts per bin, where Neyman weighting is biased low
    # (simulated with this decay: 28.6 ns instead of 30.2 ns; MLE 30.3 ns).
    assert np.max(mle["tau"]) > np.max(neyman["tau"])
    np.testing.assert_allclose(np.sort(mle["tau"]), np.sort(neyman["tau"]), rtol=0.25)
    # MLE on normalised data uses the peak to get back to counts
    norm = ptu.fit_lifetime(hist, n_exp=2, weighting="mle", normalize=True)
    np.testing.assert_allclose(np.sort(norm["tau"]), np.sort(mle["tau"]), rtol=1e-3)


def test_lifetime_trace_passes_mle_and_period_per_bin(monkeypatch):
    seen = []

    class DummyFit:
        def __init__(self, t, y, n_exp, offset, **kwargs):
            seen.append(kwargs)
            self.success = True

        def parameters(self):
            return {"A": np.ones(1), "tau": np.array([2.0]), "A_fit": np.full(1, 2.0)}

    monkeypatch.setattr("kpfm_pl.ptu.MultiExpFit", DummyFit)
    matrix = np.array([[16, 4, 1, 0, 0], [64, 16, 4, 1, 0]], dtype=np.uint32)
    p = _synthetic_ptu(matrix, bintime=1.0, acquisition_time=2.0)
    p.rep_time = 50e-9
    trace = p.lifetime_trace(bintime=1.0, mode="manual", start=-1, end=10, weighting="mle")
    assert all(k["statistic"] == "poisson_mle" and k["period"] == pytest.approx(50.0)
               for k in seen)
    assert seen[0]["count_scale"] == pytest.approx(16.0)
    np.testing.assert_allclose(trace.meta["amplitudes_fit"], 2.0)
    np.testing.assert_allclose(trace.meta["amplitudes"], 1.0)


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


# ------------------------------------------------- mean arrival time / flid_map
def _decay_ptu(tau=3.0, n_channels=200, res_ns=0.1, amplitude=1e6, background=0.0,
               rows=3):
    """Synthetic PTU with noise-free single-exponential rows (1 s bins)."""
    edges = np.arange(n_channels + 1) * res_ns
    row = amplitude * tau * (np.exp(-edges[:-1] / tau) - np.exp(-edges[1:] / tau))
    matrix = np.round(np.tile(row + background, (rows, 1))).astype(np.uint32)
    p = _synthetic_ptu(matrix, bintime=1.0, acquisition_time=float(rows))
    p.lt_time = np.arange(n_channels, dtype=float) * res_ns * 1e-9
    p.resolution = res_ns * 1e-9
    # The flat background, expressed as dark counts per second.
    p.dark_counts = background * n_channels / 1.0
    return p


def test_arrival_time_trace_corrects_truncation():
    # A 20 ns window cuts off a lot of a 5 ns decay: the raw mean is too short.
    p = _decay_ptu(tau=5.0)
    trace = p.arrival_time_trace(bintime=1.0, mode="manual", start=0, min_counts=None)
    assert trace.meta["window_ns"] == pytest.approx(20.0)
    expected_raw = 5.0 - 20.0 / np.expm1(20.0 / 5.0)
    np.testing.assert_allclose(trace.meta["mean_arrival"], expected_raw, rtol=2e-3)
    np.testing.assert_allclose(trace.y, 5.0, rtol=2e-3)
    assert trace.meta["method"] == "mean_arrival" and trace.units == {"x": "s", "y": "ns"}

    raw = p.arrival_time_trace(bintime=1.0, mode="manual", start=0, min_counts=None,
                               truncation_correction=False)
    np.testing.assert_allclose(raw.y, raw.meta["mean_arrival"])


def test_arrival_time_trace_subtracts_dark_counts():
    p = _decay_ptu(tau=3.0, background=500.0)
    trace = p.arrival_time_trace(bintime=1.0, mode="manual", start=0, min_counts=None)
    np.testing.assert_allclose(trace.y, 3.0, rtol=5e-3)
    # Without the dark-count correction the flat floor pulls the mean upwards.
    p.dark_counts = 0.0
    biased = p.arrival_time_trace(bintime=1.0, mode="manual", start=0, min_counts=None)
    assert np.all(biased.y > 3.2)


def test_arrival_time_matches_intensity_weighted_fit_lifetime():
    # Two components: the mean arrival time is sum(A tau^2) / sum(A tau).
    n, res = 2000, 0.1
    edges = np.arange(n + 1) * res
    row = sum(a * tau * (np.exp(-edges[:-1] / tau) - np.exp(-edges[1:] / tau))
              for a, tau in ((6e5, 1.0), (4e5, 6.0)))
    p = _synthetic_ptu(np.round(np.tile(row, (2, 1))).astype(np.uint32), bintime=1.0,
                       acquisition_time=2.0)
    p.lt_time = np.arange(n, dtype=float) * res * 1e-9
    p.resolution = res * 1e-9
    trace = p.arrival_time_trace(bintime=1.0, mode="manual", start=0, min_counts=None)
    expected = (0.6 * 1 + 0.4 * 36) / (0.6 * 1 + 0.4 * 6)
    np.testing.assert_allclose(trace.y, expected, rtol=2e-3)


def test_flid_mean_arrival_method_uses_flid_map():
    from kpfm_pl.ptu import flid_map

    p = _decay_ptu(tau=4.0)
    kwargs = dict(bintime=1.0, mode="manual", start=0, min_counts=None)
    image = p.flid(method="mean_arrival", lifetime_bins=10, lifetime_range=(0, 10),
                   sigma=0, **kwargs)
    direct = flid_map(p.arrival_time_trace(**kwargs), lifetime_bins=10,
                      lifetime_range=(0, 10), sigma=0)
    np.testing.assert_array_equal(image.data, direct.data)
    assert image.label == "synthetic FLID (mean arrival)"
    assert image.data[:, 4].sum() == 3  # three bins at tau = 4 ns
    with pytest.raises(ValueError, match="fitted components"):
        p.flid(method="mean_arrival", isolated=True, **kwargs)
    with pytest.raises(ValueError, match="method"):
        p.flid(method="guess", **kwargs)


def test_flid_map_redraws_a_fitted_trace_like_flid(monkeypatch):
    from kpfm_pl.ptu import flid_map

    p = _flid_ptu(monkeypatch, [2.0, 4.0, 6.0, 2.0, 4.0, 6.0])
    kwargs = dict(bintime=1.0, mode="manual", start=-1, end=10, min_counts=None)
    image = p.flid(lifetime_bins=10, intensity_bins=5, sigma=0.5, **kwargs)
    redrawn = flid_map(p.lifetime_trace(**kwargs), lifetime_bins=10, intensity_bins=5,
                       sigma=0.5)
    np.testing.assert_allclose(image.data, redrawn.data)
    assert image.label == redrawn.label == "synthetic FLID"
    assert image.meta["lifetime_range"] == redrawn.meta["lifetime_range"]
    np.testing.assert_allclose(redrawn.meta["bin_lifetimes"], [2.0, 4.0, 6.0])
    with pytest.raises(ValueError, match="counts"):
        flid_map(Signal([0.0, 1.0], [1.0, 2.0]))


# ------------------------------------------------------------ count check
def _event_ptu(rate=30.0, duration=20.0, tau=3.0, seed=0):
    """Synthetic PTU from photon events: ``rate`` photons/s, 10 MHz sync."""
    rng = np.random.default_rng(seed)
    n = rng.poisson(rate * duration)
    times = np.sort(rng.uniform(0, duration, n))
    rep = 1e-7
    p = _synthetic_ptu(np.zeros((1, 100), dtype=np.uint32), bintime=-1.0,
                       acquisition_time=duration)
    p.rep_time = rep
    # 100 channels of 0.5 ns: the decay fits well inside the 50 ns window.
    p.resolution = 5e-10
    p.lt_time = np.arange(100, dtype=float) * 5e-10
    dtime = np.minimum((rng.exponential(tau * 1e-9, n) / 5e-10).astype(int), 99)
    p.events = np.column_stack([np.floor(times / rep), dtime, times]).astype(float)
    return p


def test_count_check_warns_with_suggested_bintime():
    p = _event_ptu(rate=30.0)
    with pytest.warns(RuntimeWarning, match=r"at least 5 s"):
        trace = p.arrival_time_trace(bintime=1.0, mode="manual", start=0)
    check = trace.meta["count_check"]
    assert not check["passed"] and check["suggested_bintime"] == 5.0
    assert check["min_counts"] == 100 and check["coverage"] == 0.9
    assert trace.meta["bintime"] == 1.0

    # Median coverage and a lower threshold pass; None disables the check.
    trace = p.arrival_time_trace(bintime=1.0, mode="manual", start=0, min_counts=20,
                                 coverage=0.5)
    assert trace.meta["count_check"]["passed"]
    assert p.arrival_time_trace(bintime=1.0, mode="manual", start=0,
                                min_counts=None).meta["count_check"] is None
    with pytest.raises(ValueError, match="coverage"):
        p.arrival_time_trace(bintime=1.0, coverage=0)


def test_auto_bintime_recomputes_and_scales_the_roi_count_range(capsys):
    p = _event_ptu(rate=30.0, duration=40.0)
    roi = {"x_min": 0.0, "x_max": 40.0, "y_min": 0.0, "y_max": 1000.0}
    trace = p.arrival_time_trace(roi=roi, bintime=1.0, mode="manual", start=0,
                                 auto_bintime=True)
    assert trace.meta["bintime"] == 5.0
    assert trace.meta["count_check"]["passed"]
    assert trace.meta["count_check"]["requested_bintime"] == 1.0
    assert trace.meta["roi"]["y_max"] == 5000.0 and trace.meta["roi"]["x_max"] == 40.0
    np.testing.assert_allclose(np.diff(trace.x), 5.0)
    assert "automatic bin time 5 s" in capsys.readouterr().out


def test_auto_bintime_stops_when_the_window_is_too_short():
    p = _event_ptu(rate=3.0, duration=20.0)
    with pytest.warns(RuntimeWarning, match="too short"):
        trace = p.arrival_time_trace(bintime=1.0, mode="manual", start=0,
                                     auto_bintime=True)
    assert not trace.meta["count_check"]["passed"]


def test_fit_lifetime_returns_intensity_weighted_lifetime(ptu):
    fit = ptu.fit_lifetime(ptu.lifetime_histogram(), n_exp=2)
    A, tau = np.asarray(fit["A"]), np.asarray(fit["tau"])
    assert fit["tau_int"] == pytest.approx(np.sum(A * tau**2) / np.sum(A * tau))
    assert np.isfinite(fit["tau_int_err"]) and fit["tau_int_err"] > 0


# ------------------------------------- bins before the rise, fitted background
def _periodic_ptu(taus, amps, *, background=0.0, rows=1, period_ns=100.0, res_ns=0.1,
                  peak_ns=10.0, bintime=1.0) -> PTU:
    """Synthetic file with a periodic decay (sharp rise at ``peak_ns``) in every time bin.

    As in real files, the lifetime axis has one bin more than the period,
    which never fills.
    """
    n_period = int(round(period_ns / res_ns))
    t = np.arange(n_period) * res_ns
    since_pulse = (t - peak_ns) % period_ns
    counts = sum(a * np.exp(-since_pulse / tau) / (1 - np.exp(-period_ns / tau))
                 for a, tau in zip(amps, taus)) + background
    row = np.concatenate([counts, [0.0]])
    p = _synthetic_ptu(np.tile(row, (rows, 1)), bintime=bintime,
                       acquisition_time=rows * bintime)
    p.resolution, p.rep_time = res_ns * 1e-9, period_ns * 1e-9
    p.lt_time = np.arange(row.size) * res_ns * 1e-9
    return p


def test_fit_lifetime_prerise_fits_the_previous_pulse_tail():
    p = _periodic_ptu([4.0, 30.0], [1000.0, 200.0])
    hist = p.lifetime_histogram(p._matrix, bintime=1.0)
    plain = p.fit_lifetime(hist, n_exp=2, mode="manual", start=0.0)
    assert plain["prerise"] is False and plain["prerise_points"] == 0
    assert len(plain["t_fit"]) == 901          # tail incl. the never-filled last bin
    np.testing.assert_array_equal(plain["t_model"], plain["t_fit"])

    fit = p.fit_lifetime(hist, n_exp=2, mode="manual", start=0.0, prerise=True)
    # 80 bins lie more than 2 ns before the maximum at 10 ns; the last bin is left out
    assert fit["prerise"] is True and fit["prerise_points"] == 80
    assert fit["prerise_gap_ns"] == 2.0 and len(fit["t_fit"]) == 80 + 900
    assert fit["t_fit"][0] == pytest.approx(-10.0) and fit["left_bound_ns"] == 0.0
    # those bins are fitted at their time since the previous pulse
    np.testing.assert_allclose(fit["t_model"][:80], fit["t_fit"][:80] + 100.0)
    np.testing.assert_allclose(fit["t_model"][80:], fit["t_fit"][80:])
    order = np.argsort(fit["tau"])
    np.testing.assert_allclose(np.asarray(fit["tau"])[order], [4.0, 30.0], rtol=1e-4)
    np.testing.assert_allclose(np.asarray(fit["A"])[order], [1000.0, 200.0], rtol=1e-3)
    assert len(fit["curve"]) == len(fit["t_fit"]) == len(fit["quality"]["weighted_residuals"])
    assert fit["quality"]["n_params"] == 4 and fit["fit_background"] is False

    wide = p.fit_lifetime(hist, n_exp=2, mode="manual", start=0.0, prerise=True,
                          prerise_gap=5.0)
    assert wide["prerise_points"] == 50
    with pytest.raises(ValueError, match="before the histogram maximum"):
        p.fit_lifetime(hist, n_exp=2, mode="manual", start=0.0, prerise=True,
                       prerise_gap=20.0)
    with pytest.raises(ValueError, match="gap"):
        p.fit_lifetime(hist, n_exp=2, mode="manual", start=0.0, prerise=True,
                       prerise_gap=-1.0)


def test_fit_lifetime_fits_the_background_of_a_tail_fit():
    p = _periodic_ptu([4.0, 30.0], [1000.0, 200.0], background=7.5)
    hist = p.lifetime_histogram(p._matrix, bintime=1.0)
    for options in ({}, {"prerise": True}):
        fit = p.fit_lifetime(hist, n_exp=2, mode="manual", start=0.0, fit_background=True,
                             **options)
        assert fit["fit_background"] is True
        # The plain window keeps the never-filled last bin (as the tail fit always
        # did), whose zero pulls the background down a little; prerise drops it.
        tolerance = 1e-3 if options else 5e-2
        assert fit["background"] == pytest.approx(7.5, rel=tolerance)
        assert fit["offset"] == pytest.approx(fit["background"])
        np.testing.assert_allclose(np.sort(fit["tau"]), [4.0, 30.0], rtol=tolerance)
        assert fit["quality"]["n_params"] == 5
        assert fit["quality"]["dof"] == len(fit["t_fit"]) - 5
        # prerise leaves out the never-filled last bin; the plain window keeps it
        assert len(fit["t_fit"]) == (980 if options else 901)
    # with the offset fixed at the dark level (0) the background goes into the lifetimes
    fixed = p.fit_lifetime(hist, n_exp=2, mode="manual", start=0.0)
    assert fixed["background"] == 0.0 and fixed["background_err"] == 0.0
    assert np.max(fixed["tau"]) > 33.0


def test_fitted_background_warns_about_a_lifetime_near_the_period():
    p = _periodic_ptu([5.0, 70.0], [1000.0, 300.0], background=4.0)
    hist = p.lifetime_histogram(p._matrix, bintime=1.0)
    kwargs = dict(n_exp=2, mode="manual", start=0.0, prerise=True,
                  p0=[1000.0, 600.0, 5.0, 70.0])
    fit = p.fit_lifetime(hist, fit_background=True, **kwargs)
    assert any("laser period" in w for w in fit["warnings"])
    # also without the incomplete-decay correction, and adjustable
    plain = p.fit_lifetime(hist, fit_background=True, decay_correction=False, **kwargs)
    assert any("laser period" in w for w in plain["warnings"])
    quiet = p.fit_lifetime(hist, fit_background=True,
                           diagnostics={"period_fraction": 0.9}, **kwargs)
    assert not any("laser period" in w for w in quiet["warnings"])
    fixed = p.fit_lifetime(hist, **kwargs)
    assert not any("laser period" in w for w in fixed["warnings"])


def test_lifetime_trace_prerise_and_background_rate():
    # 1 s time bins; 200 counts per lifetime bin and second of flat background
    p = _periodic_ptu([20.0], [5000.0], background=200.0, rows=3)
    p._matrix = np.round(p._matrix).astype(np.uint32)
    common = dict(bintime=1.0, n_exp=1, mode="manual", start=0.0, min_counts=None,
                  weighting="mle")
    plain = p.lifetime_trace(**common)
    assert plain.meta["prerise"] is False and plain.meta["background_rate"] is None
    assert np.all(plain.y > 21.0)                   # the background lengthens the lifetime
    lt = p.lifetime_trace(prerise=True, background_rate=200.0, **common)
    np.testing.assert_allclose(lt.y, 20.0, rtol=5e-3)
    assert lt.meta["prerise"] is True and lt.meta["prerise_gap_ns"] == 2.0
    assert lt.meta["background_rate"] == 200.0
    # grouped lifetime bins: only whole groups before the rise are used
    grouped = p.lifetime_trace(prerise=True, background_rate=200.0, bins=4, **common)
    np.testing.assert_allclose(grouped.y, 20.0, rtol=5e-3)
    with pytest.raises(ValueError, match="background_rate"):
        p.lifetime_trace(background_rate=-1.0, **common)
    with pytest.raises(ValueError, match="before the histogram maximum"):
        p.lifetime_trace(prerise=True, prerise_gap=50.0, **common)
    image = p.flid(prerise=True, background_rate=200.0, lifetime_range=(0.0, 40.0), **common)
    assert image.meta["prerise"] is True and image.meta["background_rate"] == 200.0


def test_beyond_period_marks_unfilled_and_partial_bins():
    p = _periodic_ptu([5.0], [100.0])               # 1000-bin period, 1001 lifetime bins
    assert np.flatnonzero(p._beyond_period()).tolist() == [1000]
    p.rep_time = 99.95e-9                           # 999.5 bins: the last one is partial
    assert np.flatnonzero(p._beyond_period()).tolist() == [999, 1000]
    p.lifetime_seam_bin = 400                       # after a lifetime shift
    assert np.flatnonzero(p._beyond_period()).tolist() == [400, 1000]
    # bins after the last photon of the file never filled: the real period is shorter
    p.lifetime_seam_bin, p.rep_time = None, 100e-9
    p._raw_dtime = np.array([3.0, 995.0, 120.0])
    assert np.flatnonzero(p._beyond_period()).tolist() == [996, 997, 998, 999, 1000]
    p.lifetime_shift_ns = 10.0                      # they move along with a lifetime shift
    assert np.flatnonzero(p._beyond_period()).tolist() == [896, 897, 898, 899, 1000]


def test_fitted_background_leaves_out_never_filled_bins_of_a_real_file():
    legacy = DATA_DIR / "ptu_test1_NoLaserPulse_(360s_400nW_532nm_000).ptu"
    if not legacy.exists():
        pytest.skip("PTU test file missing")
    p = PTU(legacy)
    p.time_trace(bintime=1.0)
    hist = p.lifetime_histogram()
    last = int(np.flatnonzero(hist.y)[-1])
    assert last < p.n_lt_bins - 2                   # the header period is too long here
    fit = p.fit_lifetime(hist, n_exp=2, weighting="mle", fit_background=True, prerise=True)
    used = np.isin(np.round(fit["shifted_histogram_x"], 6), np.round(fit["t_fit"], 6))
    assert not used[last + 1:].any() and used[last]
    assert np.all(np.asarray(fit["y_fit"])[-3:] > 0)    # no empty bins at the end
