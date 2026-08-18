from pathlib import Path

import numpy as np
import pytest

from kpfm_pl.core import Image
from kpfm_pl.jpk import JPK

DATA = Path(__file__).parents[1] / "examples" / "Test data" / "jpk_test_(Drift1).jpk"

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="JPK test file missing")


@pytest.fixture(scope="module")
def jpk():
    return JPK(DATA)


def test_channel_names_include_known_channels(jpk):
    names = jpk.channel_names()
    assert "Height (measured)" in names
    assert "Precision 5" in names
    assert len(names) > 1
    assert "None" not in names


def test_thumbnail_is_kept_separate_from_measurement_channels(jpk):
    assert jpk.thumbnail is not None
    assert jpk.thumbnail.shape == (64, 64)
    assert all(data.shape == (jpk.y.size, jpk.x.size) for data in jpk.channels.values())


def test_axes_are_physical_metres(jpk):
    axes = jpk.axes()
    assert set(axes) == {"x", "y"}
    assert axes["x"].size > 0 and axes["y"].size > 0
    # calibrated scan size is on the micrometre scale
    assert 0 < float(axes["x"].max()) < 1e-3


def test_image_returns_calibrated_image(jpk):
    img = jpk.image("Precision 5", process=False)
    assert isinstance(img, Image)
    assert img.units["z"] == "V"
    assert img.units["x"] == "m"
    assert img["data"].ndim == 2
    assert img.extent is not None


def test_height_channel_is_processed(jpk):
    raw = jpk.image("Height (measured)", process=False)
    proc = jpk.image("Height (measured)", process=True)
    # processing line-levels and zeros the minimum of height channels
    assert np.nanmin(proc["data"]) == pytest.approx(0.0)
    assert not np.allclose(raw["data"], proc["data"])


def test_images_returns_dict(jpk):
    out = jpk.images(["Precision 5", "Height (measured)"])
    assert set(out) == {"Precision 5", "Height (measured)"}
    assert all(isinstance(v, Image) for v in out.values())


@pytest.mark.parametrize("channel", ["Precision 6", "Precision 6 retrace"])
def test_precision_6_processing_converts_voltage_to_force(jpk, channel):
    raw = jpk.image(channel, process=False)
    processed = jpk.image(channel, process=True)
    factor = (
        jpk.calibration["cantilever-calibration-info.sensitivity"]
        * jpk.calibration["cantilever-calibration-info.spring-constant"]
    )

    assert raw.units["z"] == "V"
    assert processed.units["z"] == "N"
    np.testing.assert_allclose(processed.data, raw.data * factor)


def test_precision_6_processing_requires_force_calibration(jpk):
    sensitivity = jpk.calibration.pop("cantilever-calibration-info.sensitivity")
    try:
        with pytest.raises(ValueError, match="requires valid"):
            jpk.image("Precision 6", process=True)
    finally:
        jpk.calibration["cantilever-calibration-info.sensitivity"] = sensitivity


def test_malformed_force_calibration_is_rejected():
    loader = JPK.__new__(JPK)
    loader.filepath = Path("malformed.jpk")
    loader.calibration = {}
    metadata = (
        "cantilever-calibration-info.sensitivity : not-a-number nm/V\n"
        "cantilever-calibration-info.spring-constant : 1.0 N/m"
    )

    with pytest.raises(ValueError, match="invalid.*sensitivity"):
        loader._read_calibration_metadata(metadata)


def test_get_save_path_creates_processed_dir(jpk, tmp_path):
    import shutil

    local = tmp_path / DATA.name
    shutil.copy(DATA, local)
    j = JPK(local)
    path = j.get_save_path("AFM")
    assert path.parent.name == "processed"
    assert path.parent.is_dir()
    assert path.suffix == ".png"
