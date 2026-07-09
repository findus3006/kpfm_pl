import numpy as np
import pytest

from kpfm_pl.zurich import Zurich

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
