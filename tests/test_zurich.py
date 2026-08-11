import numpy as np
import pytest

from kpfm_pl.core import Signal
from kpfm_pl.zurich import Zurich


def _write(tmp_path, text, name="scan.txt"):
    p = tmp_path / name
    p.write_text(text)
    return p


TWO_STREAM = """% Device dev1
% Module streamer
% Time header ignored
% StreamA, chunk info
% StreamB, chunk info
-2.0;10.0
-1.0;20.0
0.0;30.0
-2.0;1.0
-1.0;2.0
0.0;3.0
"""


def test_parse_two_streams(tmp_path):
    z = Zurich(_write(tmp_path, TWO_STREAM))
    assert z.stream_names() == ["StreamA", "StreamB"]
    # time axis is negated and flipped to forward time
    assert np.allclose(z.streams["StreamA"]["time"], [0.0, 1.0, 2.0])
    # values keep their file order
    assert np.allclose(z.streams["StreamA"]["value"], [10.0, 20.0, 30.0])
    assert np.allclose(z.streams["StreamB"]["value"], [1.0, 2.0, 3.0])
    # the reference time comes from the first stream
    assert np.allclose(z.time, [0.0, 1.0, 2.0])


def test_signal_returns_signal_with_units(tmp_path):
    z = Zurich(_write(tmp_path, TWO_STREAM))
    s = z.signal("StreamA")
    assert isinstance(s, Signal)
    assert s.units == {"x": "s", "y": "V"}
    assert s.meta["stream"] == "StreamA"
    assert np.allclose(s["y"], [10.0, 20.0, 30.0])


def test_average_bins_samples(tmp_path):
    z = Zurich(_write(tmp_path, TWO_STREAM))
    s = z.average("StreamA", bintime=2.0)  # dt=1 -> bin_size=2
    assert s.meta["averaged"] is True
    assert np.allclose(s["y"], [15.0, 30.0])


def test_missing_streams_raises(tmp_path):
    text = "% Device dev1\n-1.0;2.0\n-2.0;3.0\n"
    with pytest.raises(ValueError, match="No data streams"):
        Zurich(_write(tmp_path, text))


def test_row_count_not_divisible_raises(tmp_path):
    text = ("% StreamA, x\n% StreamB, y\n"
            "-1.0;1.0\n-2.0;2.0\n-3.0;3.0\n")  # 3 rows, 2 streams
    with pytest.raises(ValueError, match="not divisible"):
        Zurich(_write(tmp_path, text))


TRIGGER = """% TrigStream, chunk info
0.0;0.0
1.0;0.0
2.0;0.0
3.0;1.0
4.0;1.0
5.0;1.0
6.0;0.0
7.0;0.0
8.0;0.0
"""


def test_laser_triggers_threshold(tmp_path):
    z = Zurich(_write(tmp_path, TRIGGER))
    trig = z.laser_triggers("TrigStream")
    assert {"left", "right", "all"} <= set(trig)
    assert trig["left"]["index"] == 3
    assert trig["right"]["index"] == 5
    # "all" holds [index, time] rows for the two threshold edges
    assert trig["all"].shape == (2, 2)
    assert trig["all"][0, 0] == 3 and trig["all"][1, 0] == 5


def test_laser_triggers_none_when_flat(tmp_path):
    text = ("% Flat, x\n" + "".join(f"{i}.0;0.0\n" for i in range(5)))
    z = Zurich(_write(tmp_path, text))
    with pytest.warns(RuntimeWarning, match="no trigger found"):
        assert z.laser_triggers("Flat") is None


def test_laser_triggers_threshold_single_on_sample_returns_none(tmp_path):
    # only one sample crosses the 0.1 threshold -> degenerate, refuse to guess
    text = "% TrigStream, x\n" + "".join(
        f"{i}.0;{1.0 if i == 3 else 0.0}\n" for i in range(6)
    )
    z = Zurich(_write(tmp_path, text))
    with pytest.warns(RuntimeWarning, match="fewer than two on-samples"):
        assert z.laser_triggers("TrigStream") is None


def test_laser_triggers_with_reference_maps_indices(tmp_path):
    z = Zurich(_write(tmp_path, TRIGGER))
    ref = Signal(np.linspace(0.0, 8.0, 5), np.zeros(5))
    trig = z.laser_triggers("TrigStream", reference=ref)
    # indices now point into the coarse reference axis
    assert 0 <= trig["left"]["index"] < len(ref["x"])
    assert 0 <= trig["right"]["index"] < len(ref["x"])
    # the "all" array is remapped onto the reference axis too
    assert trig["all"].shape == (2, 2)
    assert all(0 <= int(i) < len(ref["x"]) for i in trig["all"][:, 0])


# Multi-pulse trigger signal: on/off/on/off -> four flanks total.
MULTI_TRIGGER = """% TrigStream, chunk info
0.0;0.0
1.0;0.0
2.0;1.0
3.0;1.0
4.0;0.0
5.0;0.0
6.0;1.0
7.0;1.0
8.0;0.0
9.0;0.0
"""


def test_laser_triggers_slope_finds_all_flanks(tmp_path):
    z = Zurich(_write(tmp_path, MULTI_TRIGGER))
    trig = z.laser_triggers("TrigStream", method="slope")
    assert trig is not None
    # four flanks: rise@2, fall@4, rise@6, fall@8 (post-transition samples)
    assert trig["all"].shape == (4, 2)
    assert list(trig["all"][:, 0]) == [2, 4, 6, 8]
    # first and last flank become left/right; times track the parsed axis
    t_axis = z.streams["TrigStream"]["time"]
    assert trig["left"]["index"] == 2
    assert trig["right"]["index"] == 8
    assert trig["left"]["time"] == float(t_axis[2])
    assert trig["right"]["time"] == float(t_axis[8])


def test_laser_triggers_slope_none_when_flat(tmp_path):
    text = ("% Flat, x\n" + "".join(f"{i}.0;0.0\n" for i in range(5)))
    z = Zurich(_write(tmp_path, text))
    with pytest.warns(RuntimeWarning, match="no slope trigger"):
        assert z.laser_triggers("Flat", method="slope") is None


def test_laser_triggers_slope_single_flank_returns_none(tmp_path):
    # only one rising edge, no falling edge -> fewer than two flanks
    text = "% TrigStream, x\n" + "".join(
        f"{i}.0;{0.0 if i < 3 else 1.0}\n" for i in range(6)
    )
    z = Zurich(_write(tmp_path, text))
    with pytest.warns(RuntimeWarning, match="fewer than two flanks"):
        assert z.laser_triggers("TrigStream", method="slope") is None


def test_laser_triggers_unknown_method_raises(tmp_path):
    z = Zurich(_write(tmp_path, TRIGGER))
    with pytest.raises(ValueError, match="Unknown method"):
        z.laser_triggers("TrigStream", method="bogus")


# --- parser-level error propagation (added alongside the error-handling work) ---
_HEADER = "% Stream A, foo\n% Stream B, bar\n"


def test_parse_raises_when_no_valid_data_rows():
    lines = (_HEADER + "not;a;valid;row\nalso bad\n").splitlines()
    with pytest.raises(ValueError, match="No valid"):
        Zurich._parse(lines)


def test_parse_raises_without_streams():
    with pytest.raises(ValueError, match="No data streams"):
        Zurich._parse(["1.0;2.0", "3.0;4.0"])


def test_parse_warns_on_malformed_rows_but_keeps_valid():
    lines = (
        _HEADER
        + "1.0;10.0\n"
        + "2.0;20.0\n"
        + "garbage;row;extra\n"  # malformed, must be skipped with a warning
        + "3.0;30.0\n"
        + "4.0;40.0\n"
    ).splitlines()
    with pytest.warns(RuntimeWarning, match="malformed"):
        streams = Zurich._parse(lines)
    assert set(streams) == {"Stream A", "Stream B"}
    # 4 valid rows / 2 streams -> 2 samples each.
    assert all(len(s["value"]) == 2 for s in streams.values())
    assert np.allclose(np.sort(streams["Stream A"]["value"]), [10.0, 20.0])
