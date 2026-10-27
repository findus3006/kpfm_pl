import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent

from kpfm_pl.plotting import enable_zoom_pan
from kpfm_pl.selector import DraggableCircle, InteractiveBoxSelector, SingleBoxSelector


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _mouse(ax, name, xdata, ydata, button=None, step=0):
    """Dispatch a synthetic mouse event at data coordinates on ``ax``."""
    fig = ax.figure
    x, y = ax.transData.transform((xdata, ydata))
    fig.canvas.callbacks.process(name, MouseEvent(name, fig.canvas, x, y, button=button, step=step))


def _drag(ax, start, stop, button):
    _mouse(ax, "button_press_event", *start, button=button)
    _mouse(ax, "motion_notify_event", *stop)
    _mouse(ax, "button_release_event", *stop, button=button)


def _image_axes():
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((100, 100)))
    fig.canvas.draw()
    return ax


@pytest.mark.parametrize("cls", [InteractiveBoxSelector, SingleBoxSelector])
def test_box_selector_draws_box_and_ignores_middle_drag(cls):
    ax = _image_axes()
    selector = cls(ax)
    enable_zoom_pan(ax)
    _drag(ax, (20, 20), (60, 60), button=1)
    box = selector.get_boxes()[0]
    assert (box["x_min"], box["x_max"]) == pytest.approx((20, 60))
    assert (box["y_min"], box["y_max"]) == pytest.approx((20, 60))

    xlim = ax.get_xlim()
    _drag(ax, (40, 40), (50, 40), button=2)
    assert ax.get_xlim() != pytest.approx(xlim)
    assert selector.get_boxes() == [box]


def test_box_handles_rescale_with_zoom():
    ax = _image_axes()
    selector = InteractiveBoxSelector(ax)
    _drag(ax, (20, 20), (60, 60), button=1)
    width = selector.boxes[0]["handles"][0].get_width()
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0 / 2, x1 / 2)
    assert selector.boxes[0]["handles"][0].get_width() == pytest.approx(width / 2)

    selector.disconnect()
    ax.set_xlim(x0, x1)
    assert selector.boxes[0]["handles"][0].get_width() == pytest.approx(width / 2)


def test_draggable_circle_only_reacts_to_left_button():
    ax = _image_axes()
    circle = DraggableCircle(ax)
    enable_zoom_pan(ax)
    _drag(ax, (50, 50), (60, 50), button=2)
    assert circle.get_params() == (None, None)

    _drag(ax, (50, 50), (60, 50), button=1)
    center, radius = circle.get_params()
    assert center == pytest.approx((50, 50))
    assert radius == pytest.approx(10)

    # right-drag inside the circle deletes it without drawing a stray new one
    _drag(ax, (50, 50), (55, 50), button=3)
    assert circle.get_params() == (None, None)
