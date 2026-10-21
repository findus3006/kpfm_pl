import numpy as np
import pytest

from kpfm_pl.core import Signal
from kpfm_pl.match import Match, _bound_time, _trigger_times


def _points(*pairs):
    return [{"index": index, "time": time} for index, time in pairs]


def _make_pair(k_y=None, t_y=None):
    length = len(k_y) if k_y is not None else len(t_y) if t_y is not None else 101
    x = np.linspace(0.0, 10.0, length)
    if k_y is None:
        k_y = np.linspace(0.0, 1.0, length)
    if t_y is None:
        t_y = np.linspace(0.0, 10.0, length)
    kpfm = Signal(x, k_y, label="kpfm", units={"x": "s", "y": "V"})
    timetrace = Signal(x, t_y, label="tt", units={"x": "s", "y": "counts"})
    return kpfm, timetrace


# --------------------------------------------------------------- helpers
def test_trigger_times_from_points():
    assert list(_trigger_times(_points((1, 2.0), (3, 8.0)), "t")) == [2.0, 8.0]


def test_trigger_times_from_single_point_or_time():
    assert list(_trigger_times({"index": 1, "time": 2.0}, "t")) == [2.0]
    assert list(_trigger_times(3.5, "t")) == [3.5]


def test_trigger_times_from_plain_times():
    assert list(_trigger_times(np.array([1.5, 2.5]), "t")) == [1.5, 2.5]


@pytest.mark.parametrize("bad", [
    [], [np.nan], "abc", None,
    # the retired left/right pair format is rejected, not silently misread
    {"left": {"index": 0, "time": 1.0}, "right": {"index": 1, "time": 2.0}},
])
def test_trigger_times_rejects_invalid(bad):
    with pytest.raises(ValueError):
        _trigger_times(bad, "t")


def test_bound_time_accepts_time_point_and_default():
    assert _bound_time(None, 4.0, "start") == 4.0
    assert _bound_time(2.5, 4.0, "start") == 2.5
    assert _bound_time({"index": 0, "time": 3.0}, 4.0, "start") == 3.0


# --------------------------------------------------------------- alignment
def test_time_shift_aligns_first_triggers():
    kpfm, tt = _make_pair()
    m = Match(kpfm, tt,
              kpfm_triggers=_points((20, 2.0), (80, 8.0)),
              timetrace_triggers=_points((25, 2.5), (85, 8.5)))
    assert m.offset == pytest.approx(0.5)
    native = m.timetrace_signal(on_grid=False)
    assert np.allclose(native["x"], tt["x"] - 0.5)
    assert np.allclose(native["y"], tt["y"])


def test_align_selects_trigger_pair():
    kpfm, tt = _make_pair()
    m = Match(kpfm, tt,
              kpfm_triggers=_points((20, 2.0), (80, 8.0)),
              timetrace_triggers=_points((25, 2.5), (90, 9.0)),
              align=-1)
    assert m.offset == pytest.approx(1.0)


@pytest.mark.parametrize("align", [2, -3, 0.5, True])
def test_invalid_align_raises(align):
    kpfm, tt = _make_pair()
    with pytest.raises(ValueError, match="align"):
        Match(kpfm, tt,
              kpfm_triggers=_points((20, 2.0), (80, 8.0)),
              timetrace_triggers=_points((20, 2.0), (80, 8.0)),
              align=align)


def test_plain_floats_align_like_points():
    kpfm, tt = _make_pair()
    m = Match(kpfm, tt, kpfm_triggers=2.0, timetrace_triggers=2.5)
    assert m.offset == pytest.approx(0.5)
    assert m.triggers == [{"index": m.triggers[0]["index"], "time": 2.0}]


# --------------------------------------------------------------- grid
def test_grid_spans_all_samples_of_both_signals():
    kpfm, tt = _make_pair()
    m = Match(kpfm, tt, kpfm_triggers=2.0, timetrace_triggers=4.0)
    # time trace shifted by 2 s -> combined span is [-2, 10]
    assert m.grid[0] == pytest.approx(-2.0)
    assert m.grid[-1] == pytest.approx(10.0)
    k_grid, t_grid = m.kpfm_signal()["y"], m.timetrace_signal()["y"]
    # each signal is NaN only where it has no data of its own
    assert np.all(np.isnan(k_grid[m.grid < -1e-9]))
    assert np.all(np.isfinite(k_grid[m.grid >= -1e-9]))
    assert np.all(np.isnan(t_grid[m.grid > 8.0 + 1e-9]))
    assert np.all(np.isfinite(t_grid[m.grid <= 8.0 + 1e-9]))


def test_grid_passes_through_aligned_trigger_at_native_spacing():
    kpfm, tt = _make_pair()
    m = Match(kpfm, tt, kpfm_triggers=2.03, timetrace_triggers=2.03)
    assert m.dt == pytest.approx(0.1)
    assert np.any(m.grid == 2.03)
    assert np.allclose(np.diff(m.grid), m.dt)


def test_matching_bin_spacing_keeps_samples_exact():
    # with dt equal to the time-trace spacing, shifted bins land on the grid
    x = np.arange(0.0, 10.0, 0.5)
    counts = np.arange(len(x), dtype=float) ** 2
    kpfm = Signal(np.linspace(0.0, 12.0, 25), np.zeros(25))
    m = Match(kpfm, Signal(x, counts), kpfm_triggers=3.0, timetrace_triggers=1.0)
    t_grid = m.timetrace_signal()["y"]
    assert np.array_equal(t_grid[np.isfinite(t_grid)], counts)


def test_explicit_dt_controls_grid_spacing():
    kpfm, tt = _make_pair()
    m = Match(kpfm, tt, kpfm_triggers=2.0, timetrace_triggers=2.0, dt=0.25)
    assert np.allclose(np.diff(m.grid), 0.25)
    assert len(m.grid) == 41


@pytest.mark.parametrize("dt", [0, -1.0, np.inf, True])
def test_invalid_dt_raises(dt):
    kpfm, tt = _make_pair()
    with pytest.raises(ValueError, match="dt"):
        Match(kpfm, tt, kpfm_triggers=2.0, timetrace_triggers=2.0, dt=dt)


# --------------------------------------------------------------- triggers
def test_all_triggers_are_mapped_onto_grid():
    kpfm, tt = _make_pair()
    m = Match(kpfm, tt,
              kpfm_triggers=_points((20, 2.03), (50, 5.0), (80, 7.97)),
              timetrace_triggers=_points((20, 2.03), (80, 7.97)),
              dt=0.2)
    assert len(m.triggers) == 3
    for point in m.triggers:
        assert point["time"] == m.grid[point["index"]]
    assert m.kpfm_signal().meta["triggers"] == m.triggers
    assert m.timetrace_signal().meta["triggers"] == m.triggers


def test_native_triggers_refer_to_native_axis():
    kpfm, tt = _make_pair()
    m = Match(kpfm, tt,
              kpfm_triggers=_points((20, 2.0), (80, 8.0)),
              timetrace_triggers=_points((25, 2.5), (85, 8.5)))
    native = m.timetrace_signal(on_grid=False)
    assert [p["index"] for p in native.meta["triggers"]] == [25, 85]
    for point in native.meta["triggers"]:
        assert point["time"] == native["x"][point["index"]]


# --------------------------------------------------------------- extract
def test_extract_between_triggers_is_inclusive():
    kpfm, tt = _make_pair()
    m = Match(kpfm, tt,
              kpfm_triggers=_points((20, 2.0), (80, 8.0)),
              timetrace_triggers=_points((20, 2.0), (80, 8.0)))
    out = m.extract(m.triggers[0], m.triggers[-1])
    assert set(out) == {"kpfm", "timetrace"}
    k = out["kpfm"]
    assert k["x"][0] == pytest.approx(2.0) and k["x"][-1] == pytest.approx(8.0)
    assert len(k["x"]) == 61
    # triggers are re-indexed onto the cropped grid
    assert [p["index"] for p in k.meta["triggers"]] == [0, 60]
    assert np.allclose(out["timetrace"]["y"], np.linspace(2.0, 8.0, 61))


def test_extract_between_times():
    kpfm, tt = _make_pair()
    m = Match(kpfm, tt, kpfm_triggers=_points((20, 2.0), (80, 8.0)),
              timetrace_triggers=_points((20, 2.0), (80, 8.0)))
    out = m.extract(3.0, 5.0)
    assert out["kpfm"]["x"][0] == pytest.approx(3.0)
    assert out["kpfm"]["x"][-1] == pytest.approx(5.0)
    assert out["kpfm"].meta["triggers"] == []


def test_extract_defaults_to_overlap():
    kpfm, tt = _make_pair()
    m = Match(kpfm, tt, kpfm_triggers=2.0, timetrace_triggers=4.0)
    out = m.extract()
    assert out["kpfm"]["x"][0] == pytest.approx(0.0)
    assert out["kpfm"]["x"][-1] == pytest.approx(8.0)
    assert np.all(np.isfinite(out["kpfm"]["y"]))
    assert np.all(np.isfinite(out["timetrace"]["y"]))


def test_extract_rejects_reversed_or_empty_window():
    kpfm, tt = _make_pair()
    m = Match(kpfm, tt, kpfm_triggers=2.0, timetrace_triggers=2.0)
    with pytest.raises(ValueError, match="precede"):
        m.extract(5.0, 3.0)
    with pytest.raises(ValueError, match="no grid samples"):
        m.extract(20.0, 30.0)


# --------------------------------------------------------------- output
def test_outputs_preserve_labels_and_units():
    kpfm, tt = _make_pair()
    m = Match(kpfm, tt, kpfm_triggers=2.0, timetrace_triggers=2.0)
    assert m.kpfm_signal().label == "kpfm"
    assert m.kpfm_signal().units == {"x": "s", "y": "V"}
    assert m.timetrace_signal().label == "tt"
    assert m.timetrace_signal().units == {"x": "s", "y": "counts"}
    assert m.timetrace_signal().meta["source"]["label"] == "tt"


def test_to_dict_structure():
    kpfm, tt = _make_pair()
    m = Match(kpfm, tt, kpfm_triggers=2.0, timetrace_triggers=2.5)
    d = m.to_dict()
    assert set(d) == {"grid", "kpfm", "timetrace", "offset", "triggers"}
    assert d["offset"] == pytest.approx(0.5)
    assert len(d["grid"]) == len(m.grid)


def test_smoothing_is_applied_before_alignment():
    x = np.linspace(0.0, 10.0, 101)
    y = np.zeros(101)
    y[50] = 9.0
    m = Match(Signal(x, y), Signal(x, y), kpfm_triggers=2.0,
              timetrace_triggers=2.0, kpfm_smoothing=3)
    assert np.max(m.kpfm_signal(on_grid=False)["y"]) == pytest.approx(3.0)
    assert np.max(m.timetrace_signal(on_grid=False)["y"]) == pytest.approx(9.0)


# --------------------------------------------------------------- validation
@pytest.mark.parametrize(
    ("x", "match"),
    [
        (np.array([]), "at least two"),
        (np.array([0.0, np.nan]), "finite"),
        (np.array([0.0, 2.0, 1.0]), "strictly increasing"),
    ],
)
def test_invalid_time_axis_raises(x, match):
    y = np.ones_like(x)
    with pytest.raises(ValueError, match=match):
        Match(Signal(x, y), Signal(np.arange(3.0), np.arange(3.0)),
              kpfm_triggers=0.0, timetrace_triggers=0.0)


def test_nonoverlapping_signals_raise():
    with pytest.raises(ValueError, match="no overlapping"):
        Match(Signal([0.0, 1.0], [1.0, 2.0]),
              Signal([3.0, 4.0], [1.0, 2.0]),
              kpfm_triggers=0.0, timetrace_triggers=0.0)


@pytest.mark.parametrize("smoothing", [0, 1.5, True])
def test_invalid_smoothing_raises(smoothing):
    kpfm, tt = _make_pair()
    with pytest.raises(ValueError, match="smoothing"):
        Match(kpfm, tt, kpfm_triggers=2.0, timetrace_triggers=2.0,
              kpfm_smoothing=smoothing)
