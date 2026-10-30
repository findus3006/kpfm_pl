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


def test_set_boxes_restores_editable_boxes_and_clears():
    ax = _image_axes()
    selector = InteractiveBoxSelector(ax)
    _drag(ax, (5, 5), (10, 10), button=1)
    # Reversed bounds are normalised; previous boxes and their patches are replaced.
    selector.set_boxes([{"x_min": 60, "x_max": 20, "y_min": 20, "y_max": 60},
                        {"x_min": 70, "x_max": 90, "y_min": 70, "y_max": 90}])
    assert selector.get_boxes() == [
        {"x_min": 20.0, "x_max": 60.0, "y_min": 20.0, "y_max": 60.0},
        {"x_min": 70.0, "x_max": 90.0, "y_min": 70.0, "y_max": 90.0},
    ]
    assert len(ax.patches) == 2 * (1 + len(InteractiveBoxSelector.HANDLE_POSITIONS))

    # A restored box can be moved by dragging its interior.
    _drag(ax, (40, 40), (45, 40), button=1)
    assert selector.get_boxes()[0]["x_min"] == pytest.approx(25)

    selector.set_boxes([])
    assert selector.get_boxes() == [] and not ax.patches


def _trace_axes():
    """A long time trace: very different x (s) and y (counts) data ranges."""
    fig, ax = plt.subplots()
    ax.plot([0, 3000], [0, 20000])
    ax.set_xlim(0, 3000)
    ax.set_ylim(0, 20000)
    fig.canvas.draw()
    return ax


def _pixel_size(ax, patch):
    (x0, y0), (x1, y1) = ax.transData.transform(
        [patch.get_xy(), (patch.get_x() + patch.get_width(), patch.get_y() + patch.get_height())])
    return abs(x1 - x0), abs(y1 - y0)


def test_box_handles_keep_their_screen_size_on_zoom():
    ax = _trace_axes()
    selector = InteractiveBoxSelector(ax)
    selector.set_boxes([{"x_min": 1000, "x_max": 2000, "y_min": 5000, "y_max": 15000}])
    handle = selector.boxes[0]["handles"][0]
    expected = 2 * InteractiveBoxSelector.HANDLE_POINTS * ax.figure.dpi / 72
    # Square on screen although x and y have very different data ranges.
    assert _pixel_size(ax, handle) == pytest.approx((expected, expected))

    width, height = handle.get_width(), handle.get_height()
    ax.set_xlim(1000, 2500)
    assert handle.get_width() == pytest.approx(width / 2)
    ax.set_ylim(5000, 10000)
    assert handle.get_height() == pytest.approx(height / 4)
    assert _pixel_size(ax, handle) == pytest.approx((expected, expected))

    selector.disconnect()
    ax.set_xlim(0, 3000)
    assert handle.get_width() == pytest.approx(width / 2)


def test_box_edges_resize_and_set_the_cursor(monkeypatch):
    from matplotlib.backend_tools import Cursors

    ax = _trace_axes()
    selector = InteractiveBoxSelector(ax)
    selector.set_boxes([{"x_min": 1000, "x_max": 2000, "y_min": 5000, "y_max": 15000}])
    cursors = []
    monkeypatch.setattr(ax.figure.canvas, "set_cursor", cursors.append)

    # Hovering over the right edge (away from its handle) shows a resize cursor.
    _mouse(ax, "motion_notify_event", 2000, 7000)
    assert cursors[-1] == Cursors.RESIZE_HORIZONTAL
    _mouse(ax, "motion_notify_event", 1500, 10000)
    assert cursors[-1] == Cursors.MOVE
    _mouse(ax, "motion_notify_event", 500, 10000)
    assert cursors[-1] == Cursors.POINTER

    # Dragging that edge resizes the box instead of moving it.
    _drag(ax, (2000, 7000), (2500, 7000), button=1)
    box = selector.get_boxes()[0]
    assert box["x_min"] == pytest.approx(1000) and box["x_max"] == pytest.approx(2500)
    assert (box["y_min"], box["y_max"]) == pytest.approx((5000, 15000))

    # The top edge resizes vertically only.
    _drag(ax, (1200, 15000), (1200, 18000), button=1)
    box = selector.get_boxes()[0]
    assert box["y_max"] == pytest.approx(18000) and box["x_max"] == pytest.approx(2500)


def test_explicit_handle_size_stays_in_data_units():
    ax = _trace_axes()
    selector = InteractiveBoxSelector(ax, handle_size=10.0)
    selector.set_boxes([{"x_min": 1000, "x_max": 2000, "y_min": 5000, "y_max": 15000}])
    handle = selector.boxes[0]["handles"][0]
    ax.set_xlim(1000, 1500)
    assert (handle.get_width(), handle.get_height()) == pytest.approx((20.0, 20.0))


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
