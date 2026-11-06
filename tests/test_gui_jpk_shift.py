"""Tests of the GUI's AFM shift workflow: Qt-free model and offscreen tab."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage

from kpfm_pl.core import Image
from kpfm_pl.gui.export import write_payload_csv
from kpfm_pl.gui.models import jpk as jpk_model
from kpfm_pl.gui.models import jpk_shift as model
from kpfm_pl.gui.models.files import load_jpk

DATA_DIR = Path(__file__).parents[1] / "examples" / "Test data"
JPK_FILE = DATA_DIR / "jpk_test_(Drift1).jpk"
needs_file = pytest.mark.skipif(not JPK_FILE.exists(), reason="JPK test file missing")

Circle = jpk_model.CircleROI


def _image(data, size=(4e-6, 4e-6), label="ch"):
    data = np.asarray(data, dtype=float)
    ny, nx = data.shape
    return Image(data, size[0] / nx * np.arange(nx), size[1] / ny * np.arange(ny),
                 label=label, units={"x": "m", "y": "m", "z": "m"})


def _blobs(shape=(160, 160), seed=0):
    rng = np.random.default_rng(seed)
    return ndimage.gaussian_filter(rng.normal(size=shape), 3)


def _handle(name, channels=("Height", "KPFM")):
    reader = SimpleNamespace(channel_names=lambda: list(channels), dx=4e-6, dy=4e-6,
                             units={})
    path = Path(name)
    return SimpleNamespace(name=path.name, stem=path.stem, path=path, reader=reader)


# ------------------------------------------------------------------ model
def test_file_labels_and_common_channels():
    handles = [_handle("a.jpk", ("Height", "KPFM", "Error")), _handle("a.jpk"),
               _handle("b.jpk", ("KPFM", "Height"))]
    assert model.file_labels(handles) == ["a.jpk", "a.jpk (2)", "b.jpk"]
    assert model.common_channels(handles) == ["Height", "KPFM"]
    assert model.common_channels([]) == []


def test_follow_reference_moves_only_unmoved_circles():
    ref = Circle(1e-6, 1e-6, 1e-7)
    moved = Circle(1.5e-6, 1e-6, 1e-7)
    circles = [ref, Circle(1e-6, 1e-6, 1e-7), moved]
    new = Circle(2e-6, 2e-6, 2e-7)
    out = model.follow_reference(circles, 0, new)
    assert out[0] == new
    assert out[1] == new                                   # was a copy: follows
    assert out[2] == Circle(1.5e-6, 1e-6, 2e-7)            # moved: keeps centre
    assert model.follow_reference(out, 0, None) == [None, None, None]
    # Rounding noise from the mouse round trip does not count as a move.
    assert model.same_circle(ref, Circle(1e-6 * (1 + 1e-15), 1e-6, 1e-7))
    assert not model.same_circle(ref, moved) and model.same_circle(None, None)


def test_resample_onto_same_and_finer_grid():
    ramp = np.add.outer(np.arange(8.0), np.arange(8.0))
    ref = _image(ramp)
    assert model.resample_onto(ref, _image(ramp)) is not None
    np.testing.assert_array_equal(model.resample_onto(ref, _image(ramp)), ramp)
    # Same scan sampled twice as finely, holding a linear function of the pixel
    # positions as they are displayed: it is reproduced exactly inside.
    fine = _image(np.zeros((16, 16)))
    fx, fy = jpk_model.pixel_centres(fine, "um")
    fine = _image(2 * fx + 3 * fy)
    out = model.resample_onto(ref, fine)
    xx, yy = jpk_model.pixel_centres(ref, "um")
    assert out.shape == ramp.shape
    np.testing.assert_allclose(out[1:-1, 1:-1], (2 * xx + 3 * yy)[1:-1, 1:-1], atol=1e-9)


def test_window_slices_and_minimum_size():
    img = _image(np.zeros((100, 100)), size=(10e-6, 10e-6))     # 0.1 µm pixels
    rows, cols = model.window_slices(img, Circle(5e-6, 5e-6, 0.5e-6), window=2.0)
    assert 15 <= rows.stop - rows.start <= 23 and 15 <= cols.stop - cols.start <= 23
    with pytest.raises(ValueError, match="pixels"):
        model.window_slices(img, Circle(5e-6, 5e-6, 0.1e-6), window=1.0)


def test_detect_shift_signs_follow_the_axes():
    base = _blobs()
    ref = _image(base[20:140, 20:140], size=(12e-6, 12e-6))    # 0.1 µm pixels
    # Content moved 2 rows down and 3 columns right = +0.3 µm in x, -0.2 µm in y.
    moved = _image(ndimage.shift(base, (2, 3), order=3)[20:140, 20:140],
                   size=(12e-6, 12e-6))
    dx, dy, peak = model.detect_shift(ref, moved)
    assert dx == pytest.approx(0.3e-6, abs=0.03e-6)
    assert dy == pytest.approx(-0.2e-6, abs=0.03e-6)
    assert peak > 0.8
    dx, dy, _ = model.detect_shift(ref, moved, circle=Circle(6e-6, 6e-6, 1e-6), window=3)
    assert dx == pytest.approx(0.3e-6, abs=0.05e-6)
    assert dy == pytest.approx(-0.2e-6, abs=0.05e-6)


def test_shift_tables_apply_and_export(tmp_path):
    base = _blobs()
    size = (12e-6, 12e-6)
    images = [{"Height": _image(base[20:140, 20:140], size)},
              {"Height": _image(ndimage.shift(base, (0, 5), order=3)[20:140, 20:140], size)}]
    handles = [_handle("ref.jpk"), _handle("later.jpk")]
    labels = model.file_labels(handles)
    results = model.detect_shifts(labels, images, 0, ["Height"])
    assert len(results) == 1 and results[0].file == "later.jpk"
    assert results[0].dx == pytest.approx(0.5e-6, abs=0.03e-6)

    ref_circle = Circle(6e-6, 6e-6, 1e-6)
    circles = [ref_circle, Circle(6e-6, 6e-6, 1e-6)]
    assert model.manual_shifts(labels, circles, 0)[0].distance == 0.0
    circles = model.apply_detected(circles, 0, results, "Height")
    assert circles[1].x == pytest.approx(6.5e-6, abs=0.03e-6)
    manual = model.manual_shifts(labels, circles, 0)
    assert manual[0].source == "circle" and manual[0].dx == pytest.approx(results[0].dx)

    rows = model.shift_rows(manual + results, "um")
    assert rows[0]["dx"] == pytest.approx(results[0].dx * 1e6)
    assert model.shift_text(rows, "um").splitlines()[0].startswith("file\tsource\tdx (µm)")

    styles = {"Height": jpk_model.ChannelStyle("gold", "nm", 1e9)}
    stats = model.stats_rows(labels, images, styles, circles, "um")
    assert [r["file"] for r in stats] == labels and stats[0]["pixels"] > 0
    assert model.stats_text(stats).startswith("file\tchannel")

    ranges = model.shared_ranges(images, ["Height"])
    lo = min(np.nanmin(im["Height"].data) for im in images)
    assert ranges["Height"][0] == pytest.approx(lo)

    payloads = model.export_payloads(handles, 0, manual + results, stats, circles,
                                     channels=["Height"], process=True)
    assert [p.part for p in payloads] == ["shifts", "circle_stats"]
    assert payloads[0].default_name == "ref_shift_shifts"
    path = write_payload_csv(tmp_path / "s.csv", payloads[0])
    text = path.read_text(encoding="utf-8")
    assert "dx_m" in text and "later.jpk" in text
    name, meta = model.figure_info(handles, 0, results, circles, channels=["Height"],
                                   process=True)
    assert name == "ref_shift" and meta["parameters"]["reference"] == "ref.jpk"


# -------------------------------------------------------------------- tab
qt = pytest.importorskip


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


def _shift_tab(window):
    from kpfm_pl.gui.tabs.jpk_shift_tab import JPKShiftTab

    return next(t for t in window.tab_widgets if isinstance(t, JPKShiftTab))


def _wait(window, qtbot, condition):
    qtbot.waitUntil(lambda: condition() and not window.runner.busy, timeout=30000)


@needs_file
def test_shift_tab_two_files(window, qtbot, tmp_path, monkeypatch):
    from matplotlib.backend_bases import MouseEvent

    tab = _shift_tab(window)
    monkeypatch.setattr(window.ctx, "open_files_dialog",
                        lambda *a, **k: [str(JPK_FILE), str(JPK_FILE)])
    tab.add_files_dialog()
    _wait(window, qtbot, lambda: len(tab.images) == 2)
    assert tab.labels == [JPK_FILE.name, f"{JPK_FILE.name} (2)"]
    channels = tab._plotted_channels()
    assert set(channels) == {"Height (measured)", "Precision 5", "Photon Counter 1"}
    assert len(tab._axes) == 2 * len(channels)

    # Circle on the reference: copied to the second file.
    tab.circle_form.set_values({"x": 0.5, "y": 0.5, "radius": 0.2})
    tab.apply_circle_fields()
    assert tab.circles[0] == Circle.from_unit((0.5, 0.5), 0.2, "um")
    assert model.same_circle(tab.circles[1], tab.circles[0])
    assert tab.stats_table.rowCount() == 2 * len(channels)
    assert tab.shift_table.rowCount() == 1                      # manual, zero shift
    assert tab.results[0].distance == 0.0

    # Drag the second file's circle by 0.1 µm to the right.
    canvas = tab.plot.canvas
    canvas.draw()
    draw_on = tab.circle_form.value("draw_on")
    ax = tab._axes[(1, draw_on)]

    def send(name, xy, button):
        x, y = ax.transData.transform(xy)
        canvas.callbacks.process(name, MouseEvent(name, canvas, x, y, button=button))

    send("button_press_event", (0.5, 0.5), 1)
    send("motion_notify_event", (0.6, 0.5), 1)
    send("button_release_event", (0.6, 0.5), 1)
    assert tab.circles[1].x == pytest.approx(0.6e-6, rel=1e-3)
    assert tab.results[0].dx == pytest.approx(0.1e-6, rel=1e-2)

    # Moving the reference keeps the moved circle where it is.
    tab.circle_form.set_values({"x": 0.4, "y": 0.5, "radius": 0.25})
    tab.apply_circle_fields()
    assert tab.circles[1].x == pytest.approx(0.6e-6, rel=1e-3)
    assert tab.circles[1].radius == pytest.approx(0.25e-6)

    # Right-click on the second file: back to the reference position.
    send("button_press_event", (0.6, 0.5), 3)
    assert model.same_circle(tab.circles[1], tab.circles[0])

    # Automatic detection: the same file twice has no shift.
    tab.detect_shift()
    _wait(window, qtbot, lambda: bool(tab.auto_results))
    assert len(tab.auto_results) == len(channels)
    for result in tab.auto_results:
        assert result.distance < 1e-9
    assert tab.shift_table.rowCount() == 1 + len(channels)
    tab.shift_form.set_value("use_channel", "Height (measured)")
    tab.apply_detected()
    assert model.same_circle(tab.circles[1], tab.circles[0]) or \
        tab.circles[1].x == pytest.approx(tab.circles[0].x, abs=1e-9)

    # Detection around the circle, too.
    tab.shift_form.set_value("region", "circle")
    tab.detect_shift()
    _wait(window, qtbot, lambda: bool(tab.auto_results) and
          tab._auto_settings[0] == "circle")

    # Exports: shift table + statistics, figure named after the reference.
    target = tmp_path / "out.csv"
    monkeypatch.setattr(window.ctx, "save_file_dialog", lambda *a, **k: str(target))
    tab.plot.export_csv_dialog()
    assert (tmp_path / "out_shifts.csv").exists()
    assert (tmp_path / "out_circle_stats.csv").exists()
    name, meta = tab.plot.figure_info()
    assert name == f"{JPK_FILE.stem}_shift"
    assert len(meta["sources"]) == 2

    # Switch the reference, then remove a file.
    tab.file_form.set_value("reference", 1)
    assert tab.ref == 1 and not tab.auto_results
    tab.file_list.setCurrentRow(0)
    tab.remove_selected_file()
    assert tab.ref == 0 and len(tab.handles) == 1
    _wait(window, qtbot, lambda: len(tab.images) == 1)
    assert not tab.detect_button.isEnabled()


@needs_file
def test_shift_tab_file_limit_and_add_from_afm_tab(window, qtbot):
    tab = _shift_tab(window)
    tab.add_current_file()                         # nothing open yet: just a warning
    assert tab.handles == []
    window.ctx.load_file("jpk", JPK_FILE)
    _wait(window, qtbot, lambda: window.ctx.state.jpk is not None)
    for _ in range(model.MAX_FILES + 1):
        tab.add_current_file()
    assert len(tab.handles) == model.MAX_FILES
    assert not tab.add_files_button.isEnabled()
    _wait(window, qtbot, lambda: len(tab.images) == model.MAX_FILES)
