import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
from matplotlib.colors import LinearSegmentedColormap

from kpfm_pl.core import Image
from kpfm_pl.plotting import (
    cmap_gold,
    cmap_spring,
    crop_roi,
    enable_zoom_pan,
    make_circle,
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


def _mouse(ax, name, xdata, ydata, button=None, key=None, step=0):
    """Dispatch a synthetic mouse event at data coordinates on ``ax``."""
    fig = ax.figure
    x, y = ax.transData.transform((xdata, ydata))
    event = MouseEvent(name, fig.canvas, x, y, button=button, key=key, step=step)
    fig.canvas.callbacks.process(name, event)


def _zoom_axes(xscale="linear", yscale="linear"):
    fig, ax = plt.subplots()
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(1, 10)
    ax.set_ylim(1, 10)
    fig.canvas.draw()
    return fig, ax


def test_enable_zoom_pan_wheel_zooms_around_cursor():
    fig, ax = _zoom_axes()
    enable_zoom_pan(ax, base_scale=2.0)
    _mouse(ax, "scroll_event", 4, 7, button="up", step=1)
    # zoom in by 2 keeps the cursor position fixed
    assert ax.get_xlim() == pytest.approx((2.5, 7.0))
    assert ax.get_ylim() == pytest.approx((4.0, 8.5))
    _mouse(ax, "scroll_event", 4, 7, button="down", step=-1)
    assert ax.get_xlim() == pytest.approx((1, 10))
    assert ax.get_ylim() == pytest.approx((1, 10))


def test_enable_zoom_pan_modifiers_restrict_axis():
    fig, ax = _zoom_axes()
    enable_zoom_pan(fig, base_scale=2.0)
    _mouse(ax, "scroll_event", 5.5, 5.5, button="up", key="shift", step=1)
    assert ax.get_ylim() == pytest.approx((1, 10))
    assert ax.get_xlim() != pytest.approx((1, 10))
    _mouse(ax, "scroll_event", 5.5, 5.5, button="up", key="control", step=1)
    assert ax.get_ylim() != pytest.approx((1, 10))


def test_enable_zoom_pan_zooms_log_axes_in_log_space():
    fig, ax = _zoom_axes(yscale="log")
    ax.set_ylim(1, 1e4)
    enable_zoom_pan(ax, base_scale=2.0)
    _mouse(ax, "scroll_event", 5, 100, button="up", step=1)
    assert ax.get_ylim() == pytest.approx((10, 1e3))


def test_enable_zoom_pan_skips_colorbar_and_other_axes():
    fig, (ax, other) = plt.subplots(1, 2)
    cbar = fig.colorbar(ax.imshow(np.arange(4.0).reshape(2, 2)))
    fig.canvas.draw()
    enable_zoom_pan(ax)
    cbar_lim, other_lim = cbar.ax.get_ylim(), other.get_xlim()
    _mouse(cbar.ax, "scroll_event", 0.5, 1.0, button="up", step=1)
    _mouse(other, "scroll_event", 0.5, 0.5, button="up", step=1)
    assert cbar.ax.get_ylim() == cbar_lim
    assert other.get_xlim() == other_lim


def test_enable_zoom_pan_middle_drag_pans_and_disconnects():
    fig, ax = _zoom_axes()
    cids = enable_zoom_pan(ax)
    _mouse(ax, "button_press_event", 5, 5, button=2)
    _mouse(ax, "motion_notify_event", 6, 5)
    _mouse(ax, "button_release_event", 6, 5, button=2)
    # dragging right by one data unit moves the view left by one unit
    # (up to pixel rounding of the event position)
    assert ax.get_xlim() == pytest.approx((0, 9), abs=0.01)
    assert ax.get_ylim() == pytest.approx((1, 10))

    for cid in cids:
        fig.canvas.mpl_disconnect(cid)
    xlim = ax.get_xlim()
    _mouse(ax, "scroll_event", 5, 5, button="up", step=1)
    assert ax.get_xlim() == xlim


def test_enable_zoom_pan_toolbar_home_restores_view():
    from matplotlib.backend_bases import NavigationToolbar2

    fig, ax = _zoom_axes()
    toolbar = fig.canvas.toolbar = NavigationToolbar2(fig.canvas)
    enable_zoom_pan(ax)
    _mouse(ax, "scroll_event", 3, 3, button="up", step=1)
    _mouse(ax, "scroll_event", 3, 3, button="up", step=1)
    assert ax.get_xlim() != pytest.approx((1, 10))
    toolbar.home()
    assert ax.get_xlim() == pytest.approx((1, 10))
    assert ax.get_ylim() == pytest.approx((1, 10))


@pytest.mark.parametrize("zoom_pan", [True, False])
def test_interactive_helpers_zoom_pan_option(zoom_pan):
    # Agg's show() returns immediately, so the blocking helpers run through.
    def _wheel_zooms(ax):
        ax.figure.canvas.draw()
        xlim = ax.get_xlim()
        _mouse(ax, "scroll_event", 5, 5, button="up", step=1)
        return ax.get_xlim() != xlim

    fig, ax = plt.subplots()
    ax.imshow(np.zeros((10, 10)))
    with pytest.warns(UserWarning, match="non-interactive"):
        assert make_circle(ax, zoom_pan=zoom_pan) is None
    assert _wheel_zooms(ax) is zoom_pan

    with pytest.warns(UserWarning, match="non-interactive"):
        cropped, bounds = crop_roi(np.zeros((10, 10)), zoom_pan=zoom_pan)
    assert bounds is None
    assert _wheel_zooms(plt.gca()) is zoom_pan
