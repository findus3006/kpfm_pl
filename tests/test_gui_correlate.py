"""Tests of the Correlate workflow: Qt-free model plus offscreen tab tests."""

import csv
from pathlib import Path

import numpy as np
import pytest

from kpfm_pl.gui.export import write_payload_csv
from kpfm_pl.gui.models import correlate as model
from kpfm_pl.gui.models.files import load_ptu, load_zurich
from kpfm_pl.gui.models.ptu import TraceParams

DATA_DIR = Path(__file__).parents[1] / "examples" / "Test data"
PTU_FILE = DATA_DIR / "test3_DR1_360s_10Hz_4.67W_Scan1_001.ptu"
TXT_FILE = DATA_DIR / "test3_JS249_02,03_DR1_10Hz_AM17_Scan1.txt"
LEGACY_PTU = DATA_DIR / "ptu_test1_NoLaserPulse_(360s_400nW_532nm_000).ptu"
LEGACY_TXT = DATA_DIR / "KPFM_test1_(JV001_DR1_25Hz_AM13_1V_400nW_532nm_Scan1).txt"

pytestmark = pytest.mark.skipif(not (PTU_FILE.exists() and TXT_FILE.exists()),
                                reason="correlation test files missing")


@pytest.fixture(scope="module")
def handles():
    return load_ptu(PTU_FILE), load_zurich(TXT_FILE)


@pytest.fixture(scope="module")
def triggers(handles):
    return model.compute_triggers(*handles, TraceParams(), model.ZurichParams())


def test_triggers_found_in_both_files(triggers):
    assert triggers.zurich_params.kpfm_stream == "Demod 6 Sample Aux In 1"
    assert triggers.zurich_params.trigger_stream == "Demod 6 Sample Aux In 2"
    assert len(triggers.kpfm_triggers) == len(triggers.timetrace_triggers) == 2
    assert model.trigger_problems(triggers) == []
    # Zurich triggers live on the averaged KPFM axis (reference=)
    for point in triggers.kpfm_triggers:
        assert triggers.kpfm.x[point["index"]] == point["time"]


def test_match_between_first_and_last_trigger(triggers, tmp_path):
    result = model.compute_match(triggers, model.MatchParams(), model.WindowParams())
    summary = model.match_summary(result)
    first = triggers.kpfm_triggers[0]["time"]
    assert summary["time_shift_s"] == pytest.approx(
        triggers.timetrace_triggers[0]["time"] - first)
    assert result.start == pytest.approx(first)
    assert summary["triggers_in_window"] == 2

    payload = model.aligned_payload(result)
    assert payload.metadata["parameters"]["match"]["align_trigger_number"] == 1
    assert len(payload.metadata["sources"]) == 2
    text = write_payload_csv(tmp_path / "aligned.csv", payload).read_text(encoding="utf-8")
    header = next(line for line in text.splitlines() if not line.startswith("#"))
    columns = next(csv.reader([header]))
    assert len(columns) == 3 and columns[0] == "x (s)"  # one shared x column


def test_extract_window_by_time_and_trigger_number(triggers):
    result = model.compute_match(triggers, model.MatchParams(), model.WindowParams())
    window = model.WindowParams(start="time", start_time=100.0, stop="trigger", stop_trigger=2)
    _, start, stop = model.extract_window(result.match, window)
    assert start == pytest.approx(100.0, abs=result.match.dt)
    assert stop == pytest.approx(result.match.triggers[1]["time"])
    with pytest.raises(ValueError, match="trigger number 5"):
        model.extract_window(result.match, model.WindowParams(start="trigger", start_trigger=5))


def test_match_rejects_missing_or_out_of_range_triggers(triggers):
    with pytest.raises(ValueError, match="trigger number 3"):
        model.compute_match(triggers, model.MatchParams(align=2), model.WindowParams())
    empty = model.TriggerResult(**{**triggers.__dict__, "kpfm_triggers": None})
    assert any("Zurich" in p for p in model.trigger_problems(empty))
    with pytest.raises(ValueError, match="at least one laser trigger"):
        model.compute_match(empty, model.MatchParams(), model.WindowParams())


def test_trigger_options_number_from_one(triggers):
    options = model.align_options(triggers)
    assert [n for n, _ in options] == [1, 2]
    k, t = triggers.kpfm_triggers[1]["time"], triggers.timetrace_triggers[1]["time"]
    assert options[1][1] == f"#2 – Zurich {k:.3f} s / PTU {t:.3f} s"
    assert model.align_options(None) == []
    # Only triggers present in both files can be aligned on.
    short = model.TriggerResult(**{**triggers.__dict__,
                                   "timetrace_triggers": triggers.timetrace_triggers[:1]})
    assert [n for n, _ in model.align_options(short)] == [1]

    result = model.compute_match(triggers, model.MatchParams(), model.WindowParams())
    window = model.window_trigger_options(result.match.triggers)
    assert [n for n, _ in window] == [1, 2]
    assert window[0][1] == f"#1 – {result.match.triggers[0]['time']:.3f} s"
    assert model.window_trigger_options(None) == []


def test_trigger_payload_keeps_own_axes(triggers, tmp_path):
    payload = model.trigger_payload(triggers)
    assert len(payload.metadata["results"]["kpfm_trigger_times_s"]) == 2
    write_payload_csv(tmp_path / "triggers.csv", payload)


def test_scaled_uses_reference_range():
    np.testing.assert_allclose(model.scaled([0.0, 5.0, 10.0], [0.0, 5.0]), [0.0, 1.0, 2.0])
    np.testing.assert_allclose(model.scaled([1.0, 3.0], [np.nan, 1.0, 2.0]), [0.0, 2.0])
    np.testing.assert_allclose(model.scaled([1.0, 2.0], [3.0, 3.0]), [0.0, 0.0])


@pytest.mark.skipif(not (LEGACY_PTU.exists() and LEGACY_TXT.exists()),
                    reason="legacy test files missing")
def test_legacy_pair_needs_both_flanks():
    ptu, zurich = load_ptu(LEGACY_PTU), load_zurich(LEGACY_TXT)
    result = model.compute_triggers(ptu, zurich, TraceParams(laser_pulse=False),
                                    model.ZurichParams(flank="both"))
    assert len(result.timetrace_triggers) == 2 and len(result.kpfm_triggers) == 2
    match = model.compute_match(result, model.MatchParams(), model.WindowParams())
    assert match.stop > match.start


# ------------------------------------------------------------------ GUI tests
@pytest.fixture
def window(qtbot, tmp_path):
    pytest.importorskip("pytestqt")
    from PySide6.QtCore import QSettings

    from kpfm_pl.gui.app import MainWindow

    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    win = MainWindow(settings=settings)
    qtbot.addWidget(win)
    yield win
    win.runner.cancel_all()
    win.runner.wait(10000)


def _correlate_tab(win):
    from kpfm_pl.gui.tabs.correlate_tab import CorrelateTab

    return next(t for t in win.tab_widgets if isinstance(t, CorrelateTab))


def test_tab_finds_triggers_aligns_and_exports(window, qtbot, handles, tmp_path, monkeypatch):
    tab = _correlate_tab(window)
    assert not tab.align_button.isEnabled()
    window.state.set("ptu", handles[0])
    window.state.set("zurich", handles[1])
    assert tab.zurich_form.value("trigger_stream") == "Demod 6 Sample Aux In 2"
    qtbot.waitUntil(lambda: tab.triggers is not None and not window.runner.busy, timeout=60000)
    assert "Ready to align" in tab.trigger_label.text()
    assert tab.trigger_plot.figure.axes
    align = tab.match_form.fields["align"]
    assert [align.itemData(i) for i in range(align.count())] == [1, 2]
    assert align.itemText(1).startswith("#2 – Zurich")
    assert tab.window_form.fields["stop_trigger"].count() == 2

    tab.run_align()
    qtbot.waitUntil(lambda: tab.result is not None and not window.runner.busy, timeout=60000)
    assert tab.result_table.rowCount() == len(model.match_summary(tab.result))

    # Changing the window re-cuts the alignment without a new job
    tab.window_form.set_values({"start": "time", "start_time": 100.0})
    assert tab.result.start == pytest.approx(100.0, abs=tab.result.match.dt)
    tab.window_form.set_values({"stop": "trigger", "stop_trigger": 2})
    assert tab.window_label.text() == ""
    assert tab.result.stop == pytest.approx(tab.result.match.triggers[1]["time"])
    # Numbers outside the list cannot be selected.
    tab.window_form.set_value("stop_trigger", 9)
    assert tab.window_form.value("stop_trigger") == 2

    target = tmp_path / "aligned.csv"
    monkeypatch.setattr(window.ctx, "save_file_dialog", lambda *a, **k: str(target))
    tab.aligned_plot.export_csv_dialog()
    assert target.read_text(encoding="utf-8").startswith("# analysis: KPFM/PL correlation")


def test_align_on_second_trigger_from_dropdown(window, qtbot, handles):
    tab = _correlate_tab(window)
    window.state.set("ptu", handles[0])
    window.state.set("zurich", handles[1])
    qtbot.waitUntil(lambda: tab.triggers is not None and not window.runner.busy, timeout=60000)
    tab.match_form.set_value("align", 2)
    tab.run_align()
    qtbot.waitUntil(lambda: tab.result is not None and not window.runner.busy, timeout=60000)
    assert tab.result.match_params.align == 1
    t = tab.triggers
    assert tab.result.match.offset == pytest.approx(
        t.timetrace_triggers[1]["time"] - t.kpfm_triggers[1]["time"])
    # Loading a new file empties the trigger lists until triggers are found again.
    window.state.set("zurich", None)
    assert tab.match_form.fields["align"].count() == 0
    assert tab.window_form.fields["start_trigger"].count() == 0


def test_legacy_preset_sets_markers_and_flank(window):
    tab = _correlate_tab(window)
    tab.preset_form.set_value("preset", "legacy")
    assert tab.ptu_form.value("laser_pulse") is False
    assert tab.zurich_form.value("flank") == "both"
    tab.preset_form.set_value("preset", "current")
    assert tab.ptu_form.value("laser_pulse") is True
    assert tab.zurich_form.value("flank") == "rising"
