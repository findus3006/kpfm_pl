import json

import numpy as np
import pytest

from kpfm_pl.core import Image, Signal
from kpfm_pl.gui.export import (
    build_metadata,
    describe_meta,
    save_figure,
    write_image_csv,
    write_signals_csv,
)


def _comments(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    meta = {}
    for line in lines:
        if not line.startswith("# "):
            break
        key, _, value = line[2:].partition(": ")
        meta[key] = value
    return meta, [line for line in lines if not line.startswith("#")]


def test_build_metadata_records_sources_parameters_and_version(tmp_path):
    meta = build_metadata("PTU time trace", sources=[tmp_path / "a.ptu"],
                          parameters={"bintime": np.float64(0.05)},
                          results={"tau": np.array([1.5, 3.0])})
    assert meta["analysis"] == "PTU time trace"
    assert meta["sources"] == [str(tmp_path / "a.ptu")]
    assert meta["parameters"] == {"bintime": 0.05}
    assert meta["results"] == {"tau": [1.5, 3.0]}
    assert meta["software"].startswith("kpfm_pl ")
    json.dumps(meta)  # fully serialisable


def test_describe_meta_drops_large_arrays_and_reduces_triggers():
    meta = {"bintime": 0.05, "matrix": np.zeros((100, 100)),
            "triggers": [{"index": 1, "time": 0.5}, {"index": 4, "time": 2.0}]}
    out = describe_meta(meta)
    assert out == {"bintime": 0.05, "trigger_times_s": [0.5, 2.0]}


def test_signals_sharing_x_get_one_x_column(tmp_path):
    x = np.arange(3.0)
    a = Signal(x, x * 2, label="a", units={"x": "s", "y": "V"}, meta={"bintime": 1})
    b = Signal(x, [np.nan, 1.0, 2.0], label="b", units={"x": "s", "y": "counts"})
    path = write_signals_csv(tmp_path / "out.csv", [a, b], build_metadata("test"))
    meta, rows = _comments(path)
    assert meta["analysis"] == "test"
    assert json.loads(meta["signal_meta"])["a"] == {"bintime": 1}
    assert rows[0] == "x (s),a y (V),b y (counts)"
    assert rows[1] == "0.0,0.0,"  # NaN is an empty cell
    data = np.genfromtxt(path, delimiter=",", skip_header=len(meta) + 1)
    assert data.shape == (3, 3)
    assert np.isnan(data[0, 2])


def test_signals_with_different_x_are_padded(tmp_path):
    a = Signal([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], label="a", units={"x": "s", "y": ""})
    b = Signal([0.0], [5.0], label="b", units={"x": "ns", "y": ""})
    _, rows = _comments(write_signals_csv(tmp_path / "o.csv", [a, b], {}))
    assert rows[0] == "a x (s),a y,b x (ns),b y"
    assert rows[2] == "1.0,2.0,,"


def test_image_csv_writes_axes_as_first_row_and_column(tmp_path):
    img = Image(np.array([[1.0, 2.0], [3.0, 4.0]]), x=[10.0, 20.0], y=[0.1, 0.2],
                label="map", units={"x": "ns", "y": "counts", "z": "occurrence"})
    meta, rows = _comments(write_image_csv(tmp_path / "img.csv", img, {"analysis": "FLID"}))
    assert meta["analysis"] == "FLID"
    assert json.loads(meta["image"])["shape"] == [2, 2]
    assert rows[0].endswith(",10.0,20.0")
    assert rows[1] == "0.1,1.0,2.0"


@pytest.mark.parametrize("suffix", [".png", ".svg", ".pdf"])
def test_save_figure_embeds_metadata(tmp_path, suffix):
    from matplotlib.figure import Figure

    fig = Figure()
    fig.add_subplot().plot([0, 1], [0, 1])
    path = save_figure(fig, tmp_path / f"fig{suffix}", build_metadata("Trace test"))
    assert path.stat().st_size > 0
    if suffix == ".svg":
        assert "Trace test" in path.read_text(encoding="utf-8")


def test_save_figure_rejects_unknown_format(tmp_path):
    from matplotlib.figure import Figure

    with pytest.raises(ValueError):
        save_figure(Figure(), tmp_path / "fig.bmp", {})
