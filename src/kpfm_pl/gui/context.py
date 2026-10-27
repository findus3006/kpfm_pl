"""Application-wide state and services handed to every tab.

:class:`AppState` holds the currently loaded files (one PTU, one Zurich and
one JPK file) and announces changes, so several tabs can work on the same
file. :class:`AppContext` bundles that state with the job runner, the log and
file dialogs that remember the last used folder.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, QSettings
from PySide6.QtCore import Signal as QtSignal
from PySide6.QtWidgets import QFileDialog, QWidget

from .logcapture import LogBus
from .models.files import KINDS, LoadedFile, load_jpk, load_ptu, load_zurich
from .workers import JobRunner

__all__ = ["AppState", "AppContext"]

_LOADERS = {"ptu": load_ptu, "zurich": load_zurich, "jpk": load_jpk}


class AppState(QObject):
    """The loaded files, keyed by kind (``"ptu"``, ``"zurich"``, ``"jpk"``).

    ``file_changed(kind, handle)`` is emitted after a file was loaded (or
    closed, with ``handle`` ``None``).
    """

    file_changed = QtSignal(str, object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._files: dict[str, LoadedFile | None] = {kind: None for kind in KINDS}

    def get(self, kind: str) -> LoadedFile | None:
        return self._files[kind]

    def set(self, kind: str, handle: LoadedFile | None) -> None:
        if kind not in self._files:
            raise KeyError(kind)
        self._files[kind] = handle
        self.file_changed.emit(kind, handle)

    @property
    def ptu(self) -> LoadedFile | None:
        return self._files["ptu"]

    @property
    def zurich(self) -> LoadedFile | None:
        return self._files["zurich"]

    @property
    def jpk(self) -> LoadedFile | None:
        return self._files["jpk"]


class AppContext:
    """Services shared by all tabs.

    Parameters
    ----------
    window :
        Parent widget for dialogs.
    settings :
        Where the last used folder is remembered. Defaults to the user's
        ``QSettings("kpfm_pl", "gui")``.
    """

    def __init__(self, window: QWidget | None = None, *, state: AppState | None = None,
                 runner: JobRunner | None = None, log: LogBus | None = None,
                 settings: QSettings | None = None) -> None:
        self.window = window
        self.state = state or AppState()
        self.runner = runner or JobRunner()
        self.log = log or LogBus()
        self.settings = settings or QSettings("kpfm_pl", "gui")

    # ---------------------------------------------------------------- dialogs
    @property
    def last_dir(self) -> str:
        return str(self.settings.value("last_dir", str(Path.home())))

    def remember_dir(self, path: str | Path) -> None:
        path = Path(path)
        self.settings.setValue("last_dir", str(path if path.is_dir() else path.parent))

    def open_file_dialog(self, title: str, file_filter: str) -> str | None:
        path, _ = QFileDialog.getOpenFileName(self.window, title, self.last_dir,
                                              f"{file_filter};;All files (*)")
        if path:
            self.remember_dir(path)
        return path or None

    def open_folder_dialog(self, title: str) -> str | None:
        path = QFileDialog.getExistingDirectory(self.window, title, self.last_dir)
        if path:
            self.remember_dir(path)
        return path or None

    def save_file_dialog(self, title: str, default_name: str, file_filter: str) -> str | None:
        start = str(Path(self.last_dir) / default_name)
        path, _ = QFileDialog.getSaveFileName(self.window, title, start, file_filter)
        if path:
            self.remember_dir(path)
        return path or None

    def show_tab(self, name: str) -> None:
        """Switch the main window to the tab class ``name`` (no-op without a window)."""
        if hasattr(self.window, "show_tab"):
            self.window.show_tab(name)

    # ---------------------------------------------------------------- loading
    def load_file(self, kind: str, path: str | Path,
                  on_done: Callable[[LoadedFile], None] | None = None):
        """Load ``path`` as ``kind`` in the background and publish it in the state."""
        loader = _LOADERS[kind]
        path = Path(path)

        def _loaded(handle: LoadedFile) -> None:
            self.state.set(kind, handle)
            self.log.info(f"Loaded {path.name}")
            if on_done is not None:
                on_done(handle)

        self.log.info(f"Loading {path.name} …")
        return self.runner.submit(f"Loading {path.name}", lambda progress: loader(path),
                                  _loaded)

    def open_and_load(self, kind: str) -> None:
        """Ask for a file of ``kind`` and load it."""
        path = self.open_file_dialog(f"Open {KINDS[kind].split(' (')[0]}", KINDS[kind])
        if path:
            self.load_file(kind, path)
