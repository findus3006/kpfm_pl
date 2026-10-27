"""Offscreen smoke tests of the main window, job runner and PTU tab."""

from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QSettings  # noqa: E402

from kpfm_pl.gui.app import MainWindow  # noqa: E402
from kpfm_pl.gui.workers import JobRunner  # noqa: E402

DATA_DIR = Path(__file__).parents[1] / "examples" / "Test data"
PULSE = DATA_DIR / "ptu_test2_LaserPulse(DR1_800nW_1200_ChargeAccu_000).ptu"


@pytest.fixture
def window(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    win = MainWindow(settings=settings)
    qtbot.addWidget(win)
    yield win
    win.runner.cancel_all()
    win.runner.wait(10000)


def _wait_idle(qtbot, runner, timeout=60000):
    qtbot.waitUntil(lambda: not runner.busy, timeout=timeout)


def test_runner_delivers_results_and_errors(qtbot):
    runner = JobRunner()
    results, errors = [], []
    runner.submit("ok", lambda progress: 41 + 1, results.append)
    runner.submit("bad", lambda progress: 1 / 0, on_error=lambda m, tb: errors.append(m))
    _wait_idle(qtbot, runner)
    assert results == [42]
    assert errors and errors[0].startswith("ZeroDivisionError")


def test_runner_cancels_through_progress(qtbot):
    import threading

    runner = JobRunner()
    started, cancelled = threading.Event(), []
    runner.job_cancelled.connect(cancelled.append)

    def job(progress):
        started.set()
        while True:
            progress(0, 1)

    handle = runner.submit("loop", job, cancellable=True)
    assert started.wait(5)
    handle.cancel()
    _wait_idle(qtbot, runner)
    assert cancelled == ["loop"]


def test_log_capture_routes_print_warnings_and_restores(monkeypatch):
    import sys
    import warnings

    from kpfm_pl.gui.logcapture import LogBus, install

    # pytest-qt's own excepthook would report the chained test exception.
    monkeypatch.setattr(sys, "excepthook", lambda *args: None)
    bus = LogBus()
    messages = []
    bus.message.connect(lambda text, level: messages.append((level, text)))
    stdout, hook = sys.stdout, sys.excepthook
    uninstall = install(bus)
    try:
        print("hello", "world")
        warnings.warn("careful", RuntimeWarning)
        try:
            raise KeyError("boom")
        except KeyError:
            sys.excepthook(*sys.exc_info())
    finally:
        uninstall()
    assert ("info", "hello world") in messages
    assert ("warning", "RuntimeWarning: careful") in messages
    assert any(level == "error" and "KeyError" in text for level, text in messages)
    assert sys.stdout is stdout and sys.excepthook is hook


def test_main_window_has_all_tabs(window):
    titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert titles[0] == "PL lifetime (PTU)"
    assert len(titles) == 5


@pytest.mark.skipif(not PULSE.exists(), reason="PTU test file missing")
def test_ptu_tab_trace_fit_and_export(window, qtbot, tmp_path, monkeypatch):
    tab = window.tab_widgets[0]
    tab.trace_form.set_value("bintime", 0.5)
    window.ctx.load_file("ptu", PULSE)
    _wait_idle(qtbot, window.runner)
    qtbot.waitUntil(lambda: tab.trace is not None and not window.runner.busy, timeout=60000)
    assert tab.trace.marker_times is not None
    assert tab.trace_plot.figure.axes

    tab.run_fit()
    qtbot.waitUntil(lambda: tab.fit is not None, timeout=60000)
    _wait_idle(qtbot, window.runner)
    assert tab.fit_table.rowCount() > 0

    target = tmp_path / "fit.csv"
    monkeypatch.setattr(window.ctx, "save_file_dialog", lambda *a, **k: str(target))
    tab.hist_plot.export_csv_dialog()
    assert target.read_text(encoding="utf-8").startswith("# analysis: PTU lifetime fit")

    png = tmp_path / "trace.png"
    monkeypatch.setattr(window.ctx, "save_file_dialog", lambda *a, **k: str(png))
    tab.trace_plot.save_figure_dialog()
    assert png.stat().st_size > 0


@pytest.mark.skipif(not PULSE.exists(), reason="PTU test file missing")
def test_ptu_tab_lifetime_trace_can_be_cancelled(window, qtbot):
    tab = window.tab_widgets[0]
    tab.trace_form.set_value("bintime", 0.5)
    window.ctx.load_file("ptu", PULSE)
    qtbot.waitUntil(lambda: tab.trace is not None and not window.runner.busy, timeout=60000)

    cancelled = []
    window.runner.job_cancelled.connect(cancelled.append)
    tab.lt_form.set_value("bintime", 0.05)   # slow enough to cancel mid-way
    tab.run_lifetime_trace()
    qtbot.waitUntil(lambda: window.cancel_button.isVisibleTo(window), timeout=5000)
    window.runner.cancel_all()
    _wait_idle(qtbot, window.runner)
    assert cancelled and tab.lifetime is None
    assert tab.lt_button.isEnabled()
