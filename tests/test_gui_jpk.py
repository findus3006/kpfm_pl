"""Tests of the GUI's JPK workflow: Qt-free model and offscreen tab."""

from pathlib import Path

import numpy as np
import pytest

from kpfm_pl.core import Image
from kpfm_pl.gui.export import write_payload_csv
from kpfm_pl.gui.models import jpk as model
from kpfm_pl.gui.models.files import load_jpk

DATA_DIR = Path(__file__).parents[1] / "examples" / "Test data"
JPK_FILE = DATA_DIR / "jpk_test_(Drift1).jpk"
needs_file = pytest.mark.skipif(not JPK_FILE.exists(), reason="JPK test file missing")


def _image(data, size=4e-6):
    data = np.asarray(data, dtype=float)
    ny, nx = data.shape
    return Image(data, size / nx * np.arange(nx), size / ny * np.arange(ny), label="ch",
                 units={"x": "m", "y": "m", "z": "m"})


# ------------------------------------------------------------------ model
def test_default_channels_and_styles():
    names = ["Height", "Height (measured)", "Precision 5", "Photon Counter 1 retrace",
             "Photon Counter 1"]
    assert model.default_channels(names) == ["Height (measured)", "Precision 5",
                                             "Photon Counter 1"]
    assert model.default_channels(["Error Signal"]) == ["Error Signal"]
    assert model.default_style("Height (measured)", "m") == model.ChannelStyle("gold", "nm", 1e9)
    assert model.default_style("Precision 5", "V") == model.ChannelStyle("spring", "mV", 1e3)
    assert model.default_style("Photon Counter 1", "counts/s").cmap == "Reds"
    assert model.unit_options("") == [("arb. units", 1.0)]


def test_pixel_centres_follow_imshow_origin_upper():
    img = _image(np.zeros((2, 4)))       # extent 0..3 µm in x, 0..2 µm in y
    xx, yy = model.pixel_centres(img, "um")
    x0, x1, y0, y1 = img.extent_um
    np.testing.assert_allclose(xx[0], x0 + (np.arange(4) + 0.5) * (x1 - x0) / 4)
    # Row 0 is drawn at the top, i.e. at the largest y.
    assert yy[0, 0] > yy[1, 0]
    np.testing.assert_allclose(yy[:, 0], [y1 - 0.25 * (y1 - y0), y1 - 0.75 * (y1 - y0)])


def test_circle_mask_selects_top_left_pixel():
    data = np.zeros((4, 4))
    img = _image(data)
    xx, yy = model.pixel_centres(img, "um")
    circle = model.CircleROI.from_unit((xx[0, 0], yy[0, 0]), 0.1, "um")
    mask = model.circle_mask(img, circle, "um")
    assert mask.sum() == 1 and mask[0, 0]
    # The same circle expressed in nm selects the same pixel.
    assert model.circle_mask(img, circle, "nm")[0, 0]


def test_circle_stats_use_display_scale():
    data = np.arange(16, dtype=float).reshape(4, 4) * 1e-9   # metres
    img = _image(data)
    styles = {"ch": model.ChannelStyle("gold", "nm", 1e9)}
    circle = model.CircleROI(x=2e-6, y=1.5e-6, radius=10e-6)   # covers everything
    (row,) = model.circle_stats({"ch": img}, styles, circle, "um")
    assert row["pixels"] == 16
    assert row["mean"] == pytest.approx(7.5)
    assert row["min"] == pytest.approx(0) and row["max"] == pytest.approx(15)
    text = model.stats_text([row], circle, "um")
    assert text.splitlines()[1].startswith("channel\tunit")


def test_circle_unit_round_trip():
    circle = model.CircleROI.from_unit((1.5, 2.0), 0.25, "um")
    assert circle.x == pytest.approx(1.5e-6) and circle.radius == pytest.approx(0.25e-6)
    center, radius = circle.in_unit("nm")
    assert center == pytest.approx((1500.0, 2000.0)) and radius == pytest.approx(250.0)


@needs_file
def test_load_images_and_payload(tmp_path):
    handle = load_jpk(JPK_FILE)
    images = model.load_images(handle, ["Precision 5", "Height (measured)"], process=True)
    assert list(images) == ["Precision 5", "Height (measured)"]
    circle = model.CircleROI(5e-7, 5e-7, 2e-7)
    styles = {n: model.default_style(n, im.units["z"]) for n, im in images.items()}
    stats = model.circle_stats(images, styles, circle)
    payload = model.image_payload(handle, images["Height (measured)"], process=True,
                                  style=styles["Height (measured)"], circle=circle,
                                  stats=stats)
    assert payload.metadata["parameters"]["channel"] == "Height (measured)"
    assert payload.metadata["results"]["circle_m"]["radius"] == pytest.approx(2e-7)
    path = write_payload_csv(tmp_path / "h.csv", payload)
    rows = [line for line in path.read_text(encoding="utf-8").splitlines()
            if not line.startswith("#")]
    assert len(rows) == 1 + images["Height (measured)"].data.shape[0]


# -------------------------------------------------------------------- tab
qt = pytest.importorskip  # alias keeps the import guards below readable


@pytest.fixture
def window(qtbot, tmp_path):
    qt("PySide6")
    qt("pytestqt")
    from PySide6.QtCore import QSettings

    from kpfm_pl.gui.app import MainWindow

    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    win = MainWindow(settings=settings)
    qtbot.addWidget(win)
    yield win
    win.runner.cancel_all()
    win.runner.wait(10000)


def _jpk_tab(window):
    from kpfm_pl.gui.tabs.jpk_tab import JPKTab

    return next(t for t in window.tab_widgets if isinstance(t, JPKTab))


def _load(window, qtbot):
    tab = _jpk_tab(window)
    window.ctx.load_file("jpk", JPK_FILE)
    qtbot.waitUntil(lambda: bool(tab.images) and not window.runner.busy, timeout=30000)
    return tab


@needs_file
def test_tab_plots_default_channels(window, qtbot):
    tab = _load(window, qtbot)
    # Default channels, in the file's channel order
    assert set(tab.images) == {"Height (measured)", "Precision 5", "Photon Counter 1"}
    assert set(tab._axes) == set(tab.images)
    # Height defaults to nm on the gold map
    assert tab.styles["Height (measured)"].unit == "nm"

    tab.set_checked_channels(["Precision 5"])
    qtbot.waitUntil(lambda: list(tab.images) == ["Precision 5"] and not window.runner.busy,
                    timeout=30000)
    assert len(tab._axes) == 1


@needs_file
def test_tab_circle_fields_mirror_and_stats(window, qtbot, tmp_path, monkeypatch):
    tab = _load(window, qtbot)
    tab.circle_form.set_values({"x": 0.5, "y": 0.5, "radius": 0.2})
    tab.apply_circle_fields()
    assert tab.circle == model.CircleROI.from_unit((0.5, 0.5), 0.2, "um")
    assert tab.stats_table.rowCount() == 3
    assert len(tab._mirrors) == 2      # dashed copies on the two other channels
    assert all(row["pixels"] > 0 for row in tab.stats)

    target = tmp_path / "jpk.csv"
    monkeypatch.setattr(window.ctx, "save_file_dialog", lambda *a, **k: str(target))
    tab.export_form.set_value("channel", "Precision 5")
    tab.plot.export_csv_dialog()
    text = target.read_text(encoding="utf-8")
    assert text.startswith("# analysis: JPK channel")
    assert '"channel": "Precision 5"' in text and "circle_m" in text

    # Switching the axes unit keeps the circle where it is.
    tab.channel_form.set_value("extent_unit", "nm")
    assert tab.circle_form.value("radius") == pytest.approx(200.0)
    tab.remove_circle()
    assert tab.circle is None and tab.stats_table.rowCount() == 0


@needs_file
def test_tab_circle_drawn_with_mouse(window, qtbot):
    from matplotlib.backend_bases import MouseEvent

    tab = _load(window, qtbot)
    canvas = tab.plot.canvas
    canvas.draw()
    ax = tab._axes[tab.circle_form.value("draw_on")]

    def send(name, xy, button):
        x, y = ax.transData.transform(xy)
        event = MouseEvent(name, canvas, x, y, button=button)
        canvas.callbacks.process(name, event)

    send("button_press_event", (0.5, 0.5), 1)
    send("motion_notify_event", (0.7, 0.5), 1)
    send("button_release_event", (0.7, 0.5), 1)
    assert tab.circle is not None
    assert tab.circle.radius == pytest.approx(0.2e-6, rel=1e-3)
    assert tab.stats_table.rowCount() == 3

    send("button_press_event", (0.5, 0.5), 3)   # right-click deletes
    assert tab.circle is None
    assert tab.stats_table.rowCount() == 0
