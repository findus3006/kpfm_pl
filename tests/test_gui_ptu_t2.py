"""GUI handling of T2 files and several detectors (synthetic files, offscreen)."""

from pathlib import Path

import numpy as np
import pytest
from ptu_writer import GENERIC_T2, PICOHARP_T2, decay_events, encode_t2, write_ptu

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QSettings  # noqa: E402

from kpfm_pl.gui.app import MainWindow  # noqa: E402
from kpfm_pl.gui.models import ptu as model  # noqa: E402
from kpfm_pl.gui.models.files import load_ptu  # noqa: E402

DATA_DIR = Path(__file__).parents[1] / "examples" / "Test data"
PULSE = DATA_DIR / "ptu_test2_LaserPulse(DR1_800nW_1200_ChargeAccu_000).ptu"


@pytest.fixture
def no_sync_file(tmp_path):
    """PicoHarp 300 T2: two detectors over 1 s, one laser-trigger marker."""
    rng = np.random.default_rng(11)
    ticks = np.sort(rng.integers(0, int(1.0 / 4e-12), 6000))
    detectors = rng.integers(1, 3, ticks.size)
    events = [(int(t), "photon", int(d)) for t, d in zip(ticks, detectors)]
    events.append((int(0.5 / 4e-12) // 16 * 16, "marker", 4))
    events.sort(key=lambda event: event[0])
    path = write_ptu(tmp_path / "picoharp_t2.ptu", encode_t2(events, "picoharp"),
                     record_type=PICOHARP_T2, global_resolution=4e-12)
    return path, detectors


@pytest.fixture
def sync_file(tmp_path):
    """MultiHarp T2 with the laser pulses on channel 0: 100 ns period, tau = 3 ns."""
    rng = np.random.default_rng(12)
    events, photons = decay_events(rng, n_pulses=40000, n_photons=40000)
    path = write_ptu(tmp_path / "multiharp_t2.ptu", encode_t2(events, "generic"),
                     record_type=GENERIC_T2, global_resolution=5e-12, acquisition_ms=4,
                     sync_rate=10_000_000)
    return path, photons


@pytest.fixture
def window(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    win = MainWindow(settings=settings)
    qtbot.addWidget(win)
    yield win
    win.runner.cancel_all()
    win.runner.wait(10000)


def _load(window, qtbot, path, bintime):
    tab = window.tab_widgets[0]
    tab.trace_form.set_value("bintime", bintime)
    previous = tab.trace
    window.ctx.load_file("ptu", path)
    qtbot.waitUntil(lambda: tab.trace is not None and tab.trace is not previous
                    and not window.runner.busy, timeout=60000)
    return tab


def _figure_text(panel) -> str:
    return " ".join(text.get_text() for text in panel.figure.texts)


# ---------------------------------------------------------------------- model
def test_model_describes_and_selects_detectors(no_sync_file, sync_file):
    handle = load_ptu(no_sync_file[0])
    text = model.describe_file(handle)
    assert "PicoHarp 300, T2 mode" in text and "detectors 1, 2" in text
    assert "no lifetimes" in text
    assert model.channel_options(None) == [("all", "All detectors")]
    assert model.channel_options(handle) == [("all", "All detectors (1 + 2)"),
                                             (1, "Detector 1"), (2, "Detector 2")]
    params = model.TraceParams(bintime=0.1)
    full = model.compute_trace(handle, params)
    assert full.has_lifetime is False and full.channels is None
    assert full.signal.y.sum() == len(no_sync_file[1])
    assert [point["time"] for point in full.triggers] == [0.5]
    assert model.trace_payload(full).metadata["results"]["detector_channels"] == "all"
    assert model.apply_lifetime_shift(handle, "auto") == 0.0        # nothing to shift

    assert model.apply_channels(handle, [2]) == (2,)
    single = model.compute_trace(handle, params)
    assert single.channels == (2,)
    assert single.signal.y.sum() == np.count_nonzero(no_sync_file[1] == 2)
    assert single.signal.meta["channels"] == [2]
    assert model.trace_payload(single).metadata["results"]["detector_channels"] == [2]
    assert model.apply_channels(handle, None) == (1, 2)

    with_sync = load_ptu(sync_file[0])
    text = model.describe_file(with_sync)
    assert "T2 mode" in text and "laser pulses on channel 0 (period 100 ns)" in text
    assert "40 ps per bin" in text


def test_model_fits_a_t2_file_with_laser_pulses(sync_file):
    handle = load_ptu(sync_file[0])
    params, states = model.TraceParams(bintime=1e-3), model.StateParams()
    model.apply_channels(handle, [1])
    result = model.compute_fit(handle, params, model.FitParams(n_exp=1), states,
                               options=model.FitOptions(weighting="mle"))
    assert result.fit["tau"][0] == pytest.approx(3.0, rel=0.06)
    assert result.fit["channels"] == [1]
    assert model.fit_summary(result.fit)["channels"] == "1"
    model.apply_channels(handle, None)
    both = model.compute_fit(handle, params, model.FitParams(n_exp=1), states)
    assert "channels" not in both.fit and "channels" not in model.fit_summary(both.fit)


# ------------------------------------------------------------------------ tab
def test_tab_shows_time_trace_only_for_a_t2_file_without_laser_pulses(window, qtbot,
                                                                      no_sync_file):
    path, detectors = no_sync_file
    tab = _load(window, qtbot, path, 0.1)
    assert "PicoHarp 300, T2 mode" in tab.info_label.text()
    assert tab.trace.signal.y.sum() == len(detectors)
    assert tab.trace_button.isEnabled()
    assert not any(b.isEnabled() for b in (tab.fit_button, tab.lt_button, tab.flid_button))
    assert not tab.shift_form.isEnabled()
    assert "no lifetime information" in _figure_text(tab.hist_plot)
    assert tab.preview_hist is None

    # the detector selector appears for two detectors and applies with the trace
    form = tab.channel_form
    assert form.isVisibleTo(tab)
    combo = form.fields["channels"]
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "All detectors (1 + 2)", "Detector 1", "Detector 2"]
    form.set_value("channels", 2)
    assert tab.channel_value() == (2,)
    previous = tab.trace
    tab.update_trace()
    qtbot.waitUntil(lambda: tab.trace is not previous and not window.runner.busy,
                    timeout=60000)
    assert tab.trace.channels == (2,)
    assert tab.trace.signal.y.sum() == np.count_nonzero(detectors == 2)
    assert not tab.fit_button.isEnabled()


def test_tab_resets_the_detector_for_a_new_file(window, qtbot, no_sync_file):
    if not PULSE.exists():
        pytest.skip("PTU test file missing")
    tab = _load(window, qtbot, no_sync_file[0], 0.1)
    tab.channel_form.set_value("channels", 1)
    previous = tab.trace
    tab.update_trace()
    qtbot.waitUntil(lambda: tab.trace is not previous and not window.runner.busy,
                    timeout=60000)
    assert tab.trace.channels == (1,)

    tab = _load(window, qtbot, PULSE, 1.0)          # PicoHarp 300 T3, one detector
    assert not tab.channel_form.isVisibleTo(tab) and tab.channel_value() is None
    assert tab.trace.channels is None and tab.trace.has_lifetime
    assert "PicoHarp 300, T3 mode" in tab.info_label.text()
    assert all(b.isEnabled() for b in (tab.fit_button, tab.lt_button, tab.flid_button))
    assert tab.shift_form.isEnabled() and tab.preview_hist is not None


def test_tab_fits_one_detector_of_a_t2_file_with_laser_pulses(window, qtbot, sync_file):
    tab = _load(window, qtbot, sync_file[0], 1e-3)
    assert tab.trace.has_lifetime and tab.fit_button.isEnabled()
    assert tab.channel_form.isVisibleTo(tab) and tab.preview_hist is not None
    tab.channel_form.set_value("channels", 1)
    messages = []
    window.ctx.log.message.connect(lambda text, level: messages.append(text))
    tab.fit_form.set_value("n_exp", 1)
    tab.run_fit()                                   # selector not applied yet: warns
    qtbot.waitUntil(lambda: tab.fit is not None and not window.runner.busy, timeout=60000)
    assert any("detector" in text for text in messages)
    assert "channels" not in tab.fit.fit

    previous = tab.trace
    tab.update_trace()
    qtbot.waitUntil(lambda: tab.trace is not previous and not window.runner.busy,
                    timeout=60000)
    tab.run_fit()
    qtbot.waitUntil(lambda: tab.fit is not None and tab.fit.fit.get("channels") == [1]
                    and not window.runner.busy, timeout=60000)
    assert tab.fit.fit["tau"][0] == pytest.approx(3.0, rel=0.1)
    rows = {tab.fit_table.item(r, 0).text(): tab.fit_table.item(r, 1).text()
            for r in range(tab.fit_table.rowCount())}
    assert rows["Detector channel(s)"] == "1"
