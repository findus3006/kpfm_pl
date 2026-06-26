import numpy as np
import pytest

from kpfm_pl.core import Image


def _image():
    return Image(
        np.zeros((4, 4)),
        x=np.linspace(0.0, 1e-5, 4),
        y=np.linspace(0.0, 2e-5, 4),
        units={"x": "m", "y": "m", "z": "V"},
    )


def test_extent_scaled_to_um():
    img = _image()
    assert img.extent_scaled("um") == (0.0, 10.0, 0.0, 20.0)


def test_extent_um_property_matches():
    img = _image()
    assert img.extent_um == img.extent_scaled("um")


def test_extent_scaled_to_nm():
    img = _image()
    assert img.extent_scaled("nm") == (0.0, 10000.0, 0.0, 20000.0)


def test_extent_scaled_unknown_unit_raises():
    with pytest.raises(ValueError):
        _image().extent_scaled("furlong")


def test_extent_scaled_none_without_axes():
    img = Image(np.zeros((2, 2)))
    assert img.extent_scaled("um") is None
