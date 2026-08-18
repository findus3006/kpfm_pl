from collections.abc import Mapping

import numpy as np
import pytest

from kpfm_pl.core import Image, Signal


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


def test_extent_scales_x_and_y_units_independently():
    img = Image(
        np.zeros((2, 2)),
        x=np.array([0.0, 1e-3]),
        y=np.array([0.0, 2.0]),
        units={"x": "m", "y": "mm", "z": "V"},
    )
    assert img.extent_scaled("um") == (0.0, 1000.0, 0.0, 2000.0)


def test_signal_validates_shape_and_length():
    with pytest.raises(ValueError, match="1-D"):
        Signal(np.zeros((2, 2)), np.zeros(4))
    with pytest.raises(ValueError, match="equal lengths"):
        Signal(np.zeros(2), np.zeros(3))


def test_image_validates_data_and_axis_shapes():
    with pytest.raises(ValueError, match="2-D"):
        Image(np.zeros(4))
    with pytest.raises(ValueError, match="x length"):
        Image(np.zeros((2, 3)), x=np.zeros(2), y=np.zeros(2))
    with pytest.raises(ValueError, match="y length"):
        Image(np.zeros((2, 3)), x=np.zeros(3), y=np.zeros(3))


@pytest.mark.parametrize(
    ("factory", "units"),
    [
        (lambda units: Signal(np.zeros(2), np.zeros(2), units=units), {"x": "s"}),
        (lambda units: Image(np.zeros((2, 2)), units=units), {"x": "m", "y": "m"}),
    ],
)
def test_containers_validate_unit_keys(factory, units):
    with pytest.raises(ValueError, match="units must contain exactly"):
        factory(units)


@pytest.mark.parametrize("container", [Signal(np.arange(2), np.arange(2)), _image()])
def test_containers_offer_dict_style_access(container):
    assert container.get("label") == container.label
    assert container.get("missing", "fallback") == "fallback"
    values = dict(container.items())
    assert tuple(values) == container.keys()
    assert values[container.keys()[0]] is container[container.keys()[0]]
    assert tuple(dict(container)) == container.keys()
    with pytest.raises(KeyError):
        container["missing"]


def test_signal_len_remains_sample_count_for_compatibility():
    signal = Signal(np.arange(3), np.arange(3))
    assert len(signal) == 3
    assert not isinstance(signal, Mapping)
