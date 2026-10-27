"""Main window and application entry point."""

from __future__ import annotations

import html
import os
import sys
import warnings

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTabWidget,
)

from .. import __version__
from .context import AppContext, AppState
from .logcapture import LogBus, install
from .tabs.correlate_tab import CorrelateTab
from .tabs.database_tab import DatabaseTab
from .tabs.flid_tab import FlidTab
from .tabs.jpk_tab import JPKTab
from .tabs.ptu_tab import PTUTab
from .workers import JobRunner

__all__ = ["MainWindow", "run"]

#: Tabs in display order.
TAB_CLASSES = (PTUTab, FlidTab, CorrelateTab, JPKTab, DatabaseTab)

_LEVEL_COLOURS = {"info": None, "warning": "#b36b00", "error": "#c00000"}

_GUIDE = """<h3>kpfm_pl quick guide</h3>
<p>Each tab is one workflow. Controls are on the left, plots on the right.
Hover over any field for an explanation.</p>
<ul>
<li><b>PL lifetime (PTU)</b>: time trace, on/off states, lifetime fit and
lifetime trace of one .ptu file.</li>
<li><b>FLID</b>: draw regions on the time trace and compare their
fluorescence-lifetime-intensity distributions.</li>
<li><b>Correlate</b>: align a KPFM signal (.txt) with a PL time trace (.ptu)
on their laser triggers.</li>
<li><b>AFM (JPK)</b>: view calibrated AFM channels.</li>
<li><b>Database</b>: index a measurement folder and open pairs from it.</li>
</ul>
<p><b>Plots:</b> mouse wheel zooms (Shift: x only, Ctrl: y only), middle-drag
pans, the Home button resets the view. <i>Save figure</i> writes PNG/SVG/PDF,
<i>Export data</i> writes CSV. Both include the analysis parameters.</p>
<p>Long computations run in the background; the status bar shows their progress
and the <i>Cancel</i> button stops them. Messages and warnings appear in the
log at the bottom.</p>"""


class MainWindow(QMainWindow):
    """The kpfm_pl main window: workflow tabs, log panel and status bar."""

    def __init__(self, settings: QSettings | None = None) -> None:
        super().__init__()
        self.setWindowTitle(f"kpfm_pl {__version__}")
        self.resize(1400, 900)

        self.log_bus = LogBus(self)
        self.runner = JobRunner(self)
        self.state = AppState(self)
        self.ctx = AppContext(self, state=self.state, runner=self.runner, log=self.log_bus,
                              settings=settings)

        self.tabs = QTabWidget()
        self.tab_widgets = [cls(self.ctx) for cls in TAB_CLASSES]
        for tab in self.tab_widgets:
            self.tabs.addTab(tab, tab.title)
        self.setCentralWidget(self.tabs)

        self._build_log()
        self._build_status_bar()
        self._build_menus()

        self.log_bus.message.connect(self.append_log)
        self.runner.busy_changed.connect(self._busy_changed)
        self.runner.progress.connect(self._progress)
        self.runner.job_started.connect(self._job_started)
        self.runner.job_failed.connect(self._job_failed)
        self.runner.job_cancelled.connect(
            lambda title: self.log_bus.warning(f"Cancelled: {title}"))
        self.state.file_changed.connect(self._file_changed)

    # ---------------------------------------------------------------- layout
    def _build_log(self) -> None:
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        dock = QDockWidget("Log", self)
        dock.setObjectName("log")
        dock.setWidget(self.log_view)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable
                         | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self.resizeDocks([dock], [140], Qt.Orientation.Vertical)

    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        self.files_label = QLabel("No files loaded")
        self.status_label = QLabel("")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(220)
        self.progress_bar.setVisible(False)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setToolTip("Stop the running computation.")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self.runner.cancel_all)
        bar.addWidget(self.files_label, 1)
        bar.addPermanentWidget(self.status_label)
        bar.addPermanentWidget(self.progress_bar)
        bar.addPermanentWidget(self.cancel_button)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for kind, text, shortcut in (("ptu", "Open &PTU file…", "Ctrl+O"),
                                     ("zurich", "Open &Zurich file…", "Ctrl+K"),
                                     ("jpk", "Open &JPK file…", "Ctrl+J")):
            action = QAction(text, self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(lambda _=False, k=kind: self.ctx.open_and_load(k))
            file_menu.addAction(action)
        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        help_menu = self.menuBar().addMenu("&Help")
        guide = QAction("Quick &guide", self)
        guide.setShortcut(QKeySequence.StandardKey.HelpContents)
        guide.triggered.connect(lambda: QMessageBox.information(self, "Quick guide", _GUIDE))
        help_menu.addAction(guide)
        about = QAction("&About", self)
        about.triggered.connect(lambda: QMessageBox.about(
            self, "About kpfm_pl",
            f"kpfm_pl {__version__}\nKPFM, photoluminescence and AFM analysis."))
        help_menu.addAction(about)

    def show_tab(self, name: str) -> None:
        """Bring the tab whose class is called ``name`` (e.g. ``"CorrelateTab"``) to front."""
        for tab in self.tab_widgets:
            if type(tab).__name__ == name:
                self.tabs.setCurrentWidget(tab)
                return

    # ---------------------------------------------------------------- slots
    def append_log(self, text: str, level: str = "info") -> None:
        colour = _LEVEL_COLOURS.get(level)
        escaped = html.escape(text)
        self.log_view.appendHtml(f'<span style="color:{colour}">{escaped}</span>'
                                 if colour else escaped)

    def _job_started(self, title: str) -> None:
        self.status_label.setText(f"{title} …")
        self.cancel_button.setVisible(
            any(job.cancellable for job in self.runner.active_jobs()))

    def _busy_changed(self, busy: bool) -> None:
        self.progress_bar.setVisible(busy)
        self.progress_bar.setRange(0, 0)  # indeterminate until progress arrives
        if not busy:
            self.cancel_button.setVisible(False)
            self.status_label.setText("")

    def _progress(self, done: int, total: int, title: str) -> None:
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(done)
        self.status_label.setText(f"{title}: {done}/{total}")

    def _job_failed(self, title: str, message: str, tb: str) -> None:
        self.log_bus.error(f"{title} failed: {message}")
        for line in tb.strip().splitlines()[-6:]:
            self.log_bus.error("    " + line)
        QMessageBox.warning(self, title, f"{title} failed:\n\n{message}")

    def _file_changed(self, kind: str, handle) -> None:
        parts = [f"{k.upper()}: {self.state.get(k).name}"
                 for k in ("ptu", "zurich", "jpk") if self.state.get(k) is not None]
        self.files_label.setText("   |   ".join(parts) if parts else "No files loaded")

    def closeEvent(self, event) -> None:
        self.runner.cancel_all()
        self.runner.wait(5000)
        super().closeEvent(event)


def run(argv: list[str] | None = None) -> int:
    """Start the GUI and block until the window is closed; return the exit code."""
    os.environ.setdefault("QT_API", "pyside6")
    import matplotlib

    matplotlib.use("QtAgg")
    app = QApplication.instance() or QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("kpfm_pl")
    app.setOrganizationName("kpfm_pl")
    window = MainWindow()
    uninstall = install(window.log_bus)
    # Show every warning, not just the first one per code location.
    warnings.simplefilter("always")
    window.show()
    window.append_log(f"kpfm_pl {__version__} ready. Open a file via the File menu "
                      "or the tab's 'Open' button.")
    try:
        return app.exec()
    finally:
        uninstall()
