import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from kpfm_pl.core import Image
from kpfm_pl.plotting import (
    cmap_gold,
    cmap_spring,
    plot_image,
    set_cbar_formatter,
    set_scaled_rcparams,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _make_image():
    data = np.arange(12.0).reshape(3, 4)
    x = np.linspace(0.0, 3e-6, 4)
    y = np.linspace(0.0, 2e-6, 3)
    return Image(data, x, y, label="Height", units={"x": "m", "y": "m", "z": "m"})


def test_colormaps_are_segmented_and_span_range():
    for cmap in (cmap_spring, cmap_gold):
        assert isinstance(cmap, LinearSegmentedColormap)
        # endpoints are (near) black at 0 and light at 1
        assert cmap(0.0)[:3] == pytest.approx((0.0, 0.0, 0.0))
        assert sum(cmap(1.0)[:3]) > sum(cmap(0.0)[:3])


def test_set_scaled_rcparams_scales_fonts():
    set_scaled_rcparams((18, 12), base=(9, 6))
    # doubling the figure size doubles the tuned title size (12 -> 24)
    assert plt.rcParams["axes.titlesize"] == pytest.approx(24.0)
    set_scaled_rcparams((9, 6), base=(9, 6))
    assert plt.rcParams["axes.titlesize"] == pytest.approx(12.0)


def test_plot_image_returns_artists_and_extent():
    img = _make_image()
    fig, ax, im, cbar = plot_image(img, cmap=cmap_gold, cbar_scale=1e9,
                                   extent_unit="um")
    # extent is rescaled from metres to micrometres
    assert im.get_extent() == pytest.approx((0.0, 3.0, 0.0, 2.0))
    assert ax.get_title() == "Height"
    assert ax.get_xlabel() == "x (um)"
    assert cbar is not None


def test_plot_image_into_existing_axis():
    img = _make_image()
    fig, ax = plt.subplots()
    fig2, ax2, im, cbar = plot_image(img, ax=ax)
    assert ax2 is ax
    assert fig2 is fig


def test_set_cbar_formatter_scales_ticks():
    img = _make_image()
    fig, ax, im, cbar = plot_image(img)
    set_cbar_formatter(cbar, img, scale=1e9, threshold=5.0)
    fmt = cbar.ax.yaxis.get_major_formatter()
    # a raw value of 2 with scale 1e9 formats as the scaled integer
    assert fmt(2.0, 0) == "2000000000"


def test_set_cbar_formatter_accepts_plain_array():
    fig, ax = plt.subplots()
    im = ax.imshow(np.arange(4.0).reshape(2, 2))
    cbar = fig.colorbar(im)
    # small span -> uses the more-precise small format
    set_cbar_formatter(cbar, np.array([0.0, 1.0]), scale=1.0, threshold=5.0)
    fmt = cbar.ax.yaxis.get_major_formatter()
    assert fmt(0.5, 0) == "0.5"
