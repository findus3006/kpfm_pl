"""Antibunching tab and its model (synthetic T2 files, offscreen)."""

from pathlib import Path

import numpy as np
import pytest
from ptu_writer import GENERIC_T2, PICOHARP_T2, antibunching_events, encode_t2, write_ptu

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QSettings  # noqa: E402

from kpfm_pl.gui.app import MainWindow  # noqa: E402
from kpfm_pl.gui.export import write_payload_csv  # noqa: E402
from kpfm_pl.gui.models import g2 as model  # noqa: E402
from kpfm_pl.gui.models.files import load_ptu  # noqa: E402
from kpfm_pl.gui.tabs.g2_tab import G2Tab  # noqa: E402

DATA_DIR = Path(__file__).parents[1] / "examples" / "Test data"
PULSE = DATA_DIR / "ptu_test2_LaserPulse(DR1_800nW_1200_ChargeAccu_000).ptu"
TICK, PERIOD = 1e-11, 10000


@pytest.fixture
def emitter_file(tmp_path):
    """MultiHarp T2, laser sync on channel 0: a blinking single emitter, 40 ms."""
    rng = np.random.default_rng(21)
    n_pulses = 400000
    dark = (np.arange(n_pulses) // 50000) % 2 == 1               # off every other 5 ms
    events, _ = antibunching_events(rng, period=PERIOD, n_pulses=n_pulses, brightness=0.2,
                                    background=0.004, dark_pulses=dark)
    return write_ptu(tmp_path / "emitter.ptu", encode_t2(events, "generic", 2),
                     record_type=GENERIC_T2, global_resolution=TICK, acquisition_ms=40,
                     sync_rate=10_000_000)


@pytest.fixture
def picoharp_file(tmp_path):
    rng = np.random.default_rng(22)
    events, _ = antibunching_events(rng, period=PERIOD, n_pulses=100000, brightness=0.1,
                                    sync=False)
    return write_ptu(tmp_path / "picoharp.ptu", encode_t2(events, "picoharp"),
                     record_type=PICOHARP_T2, global_resolution=TICK, acquisition_ms=10)


@pytest.fixture
def window(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    win = MainWindow(settings=settings)
    qtbot.addWidget(win)
    yield win
    win.runner.cancel_all()
    win.runner.wait(10000)


def _tab(window) -> G2Tab:
    return next(t for t in window.tab_widgets if isinstance(t, G2Tab))


def _load(window, qtbot, path, bintime=1e-3) -> G2Tab:
    tab = _tab(window)
    tab.trace_form.set_value("bintime", bintime)
    window.ctx.load_file("ptu", path)
    qtbot.waitUntil(lambda: tab.trace is not None and not window.runner.busy, timeout=60000)
    return tab


def _compute(window, qtbot, tab):
    previous = tab.result
    tab.run_g2()
    qtbot.waitUntil(lambda: tab.result is not previous and not window.runner.busy,
                    timeout=60000)
    return tab.result


# ---------------------------------------------------------------------- model
def test_model_regions_and_availability(emitter_file, picoharp_file):
    handle = load_ptu(emitter_file)
    assert model.unavailable_reason(handle) is None
    assert "laser period 100 ns from the sync" in model.describe_g2_file(handle)
    assert "enter the laser period" in model.describe_g2_file(load_ptu(picoharp_file))
    trace = model.compute_detector_trace(handle, model.G2TraceParams(bintime=1e-3))
    assert trace.y.size == 40
    # the on state: at least 60 % of the largest count in the window
    bounds, name = model.selection(model.G2Params(region="on"), trace, [])
    assert bounds["y_min"] == pytest.approx(0.6 * trace.y.max()) and "on state" in name
    roi = {"x_min": 0.0, "x_max": 0.005, "y_min": 0.0, "y_max": 1e9}
    bounds, name = model.selection(model.G2Params(region="roi", roi=0), trace, [roi])
    assert bounds == roi and name == "ROI 1"
    with pytest.raises(ValueError, match="does not exist"):
        model.selection(model.G2Params(region="roi", roi=1), trace, [roi])
    assert [v for v, _ in model.region_options([roi])] == ["all", "on", "roi:1"]
    assert model.G2Params(period_ns=50.0, delay_ns=-2.0).g2_kwargs() == \
        pytest.approx({"period": 5e-8, "delay": -2e-9})
    with pytest.raises(ValueError, match="threshold"):
        model.G2Params(on_percent=0).validate()


def test_model_compute_and_export(emitter_file, tmp_path):
    handle = load_ptu(emitter_file)
    result = model.compute_g2(handle, model.G2TraceParams(bintime=1e-3),
                              model.G2Params(region="on"))
    assert result.signal.meta["g2_0"] < 0.2 and result.signal.meta["method"] == "pulsed"
    rows = dict((label, value) for label, value, _ in model.result_rows(result))
    assert "±" in rows["g2(0)"] and rows["Method"].startswith("pulsed")
    payload = model.g2_payload(result)
    path = write_payload_csv(tmp_path / "g2.csv", payload)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# analysis: PTU second-order correlation g2")
    assert "g2_0" in text and "x (ns)" in text
    # the lifetime state of the shared reader is untouched
    assert handle.reader.selected_channels is None and handle.reader._tt_cache == {}


def test_t3_files_are_refused():
    if not PULSE.exists():
        pytest.skip("PTU test file missing")
    reason = model.unavailable_reason(load_ptu(PULSE))
    assert "T3 file" in reason and "T2" in reason


# ---------------------------------------------------------------------- tab
def test_tab_computes_g2_of_the_on_state_and_an_roi(window, qtbot, emitter_file):
    tab = _load(window, qtbot, emitter_file)
    signal, params = tab.trace
    assert params.channels() == (1, 2) and signal.y.size == 40
    assert "laser period" in tab.info_label.text()
    tab.g2_form.set_value("region", "on")
    assert tab.g2_form.fields["on_percent"].isVisibleTo(tab)
    result = _compute(window, qtbot, tab)
    assert result.region_name.startswith("on state") and result.signal.meta["g2_0"] < 0.2
    rows = {tab.result_table.item(r, 0).text(): tab.result_table.item(r, 1).text()
            for r in range(tab.result_table.rowCount())}
    assert "±" in rows["g2(0)"] and rows["Laser period"] == "100 ns"
    ax = tab.g2_plot.figure.axes[0]
    assert "g2(0) = " in ax.get_title()
    assert {p.get_label() for p in ax.patches} >= {"Centre peak", "Side peaks"}
    tab.g2_form.set_value("plot", "normalised")
    assert "normalised" in tab.g2_plot.figure.axes[0].get_ylabel()

    tab.set_rois([{"x_min": 0.0, "x_max": 0.005, "y_min": 0.0, "y_max": 1e9}])
    tab.g2_form.set_value("region", "roi:1")
    assert not tab.g2_form.fields["tstart"].isVisibleTo(tab)
    result = _compute(window, qtbot, tab)
    assert result.region_name == "ROI 1" and result.signal.meta["selected_bins"] == 5


def test_tab_needs_two_detectors_of_a_t2_file(window, qtbot):
    if not PULSE.exists():
        pytest.skip("PTU test file missing")
    tab = _tab(window)
    window.ctx.load_file("ptu", PULSE)
    qtbot.waitUntil(lambda: window.ctx.state.ptu is not None and not window.runner.busy,
                    timeout=60000)
    assert not tab.g2_button.isEnabled() and tab.trace is None
    assert "T3 file" in " ".join(t.get_text() for t in tab.g2_plot.figure.texts)


def test_tab_uses_an_entered_period_for_picoharp_files(window, qtbot, picoharp_file):
    tab = _load(window, qtbot, picoharp_file)
    result = _compute(window, qtbot, tab)
    assert result.signal.meta["method"] == "continuous"
    tab.g2_form.set_value("period_ns", 100.0)
    result = _compute(window, qtbot, tab)
    assert result.signal.meta["method"] == "pulsed" and result.signal.meta["g2_0"] < 0.2
