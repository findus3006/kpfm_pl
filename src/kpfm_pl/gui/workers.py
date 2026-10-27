"""Run slow backend calls in background threads.

Loading a large file or fitting a FLID takes seconds to minutes, so the GUI
hands such work to a :class:`JobRunner`. A job function receives one argument,
``progress(done, total)``, which it may pass on to backend methods that accept
a ``progress`` callback (``PTU.flid``, ``PTU.lifetime_trace``). Calling it
after :meth:`JobHandle.cancel` raises :class:`Cancelled`, which aborts the job.

Results and errors are delivered back on the GUI thread through Qt signals.
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool
from PySide6.QtCore import Signal as QtSignal

__all__ = ["Cancelled", "JobHandle", "JobRunner"]


class Cancelled(Exception):
    """Raised inside a job when the user cancelled it."""


class _JobSignals(QObject):
    finished = QtSignal(object)
    failed = QtSignal(str, str)          # message, traceback
    cancelled = QtSignal()
    progress = QtSignal(int, int)


class JobHandle:
    """Reference to a submitted job; use it to cancel or query it."""

    def __init__(self, title: str, cancellable: bool) -> None:
        self.title = title
        self.cancellable = cancellable
        self._cancel = threading.Event()
        self.done = False

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()


class _Job(QRunnable):
    def __init__(self, fn: Callable[[Callable[[int, int], None]], Any],
                 handle: JobHandle, signals: _JobSignals) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self._fn = fn
        self._handle = handle
        self._signals = signals

    def _progress(self, done: int, total: int) -> None:
        if self._handle.cancel_requested:
            raise Cancelled
        self._signals.progress.emit(int(done), int(total))

    def run(self) -> None:
        try:
            if self._handle.cancel_requested:
                raise Cancelled
            result = self._fn(self._progress)
        except Cancelled:
            self._signals.cancelled.emit()
        except Exception as exc:
            self._signals.failed.emit(f"{type(exc).__name__}: {exc}", traceback.format_exc())
        else:
            self._signals.finished.emit(result)


class JobRunner(QObject):
    """Thread pool plus bookkeeping of the jobs currently running.

    Signals
    -------
    busy_changed(bool)
        ``True`` while at least one job runs.
    progress(int, int, str)
        ``done``, ``total`` and the title of the reporting job.
    job_started(str) / job_failed(str, str, str) / job_cancelled(str)
        Title (and for failures message and traceback) of a job.
    """

    busy_changed = QtSignal(bool)
    progress = QtSignal(int, int, str)
    job_started = QtSignal(str)
    job_failed = QtSignal(str, str, str)
    job_cancelled = QtSignal(str)

    def __init__(self, parent: QObject | None = None, max_threads: int = 2) -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(max_threads)
        self._active: dict[JobHandle, _JobSignals] = {}

    @property
    def busy(self) -> bool:
        return bool(self._active)

    def active_jobs(self) -> list[JobHandle]:
        return list(self._active)

    def submit(self, title: str, fn: Callable[[Callable[[int, int], None]], Any],
               on_done: Callable[[Any], None] | None = None, *,
               on_error: Callable[[str, str], None] | None = None,
               on_finally: Callable[[], None] | None = None,
               cancellable: bool = False) -> JobHandle:
        """Run ``fn(progress)`` in the background.

        ``on_done(result)``, ``on_error(message, traceback)`` and
        ``on_finally()`` run on the GUI thread. ``on_finally`` is called after
        success, failure and cancellation alike, e.g. to re-enable a button.
        Failures are additionally broadcast via ``job_failed``.
        """
        handle = JobHandle(title, cancellable)
        signals = _JobSignals()  # created on the GUI thread -> queued delivery

        def _finish() -> None:
            handle.done = True
            self._active.pop(handle, None)
            if on_finally is not None:
                on_finally()
            if not self._active:
                self.busy_changed.emit(False)

        def _done(result: Any) -> None:
            try:
                if on_done is not None:
                    on_done(result)
            finally:
                _finish()

        def _failed(message: str, tb: str) -> None:
            try:
                self.job_failed.emit(title, message, tb)
                if on_error is not None:
                    on_error(message, tb)
            finally:
                _finish()

        def _cancelled() -> None:
            try:
                self.job_cancelled.emit(title)
            finally:
                _finish()

        signals.finished.connect(_done)
        signals.failed.connect(_failed)
        signals.cancelled.connect(_cancelled)
        signals.progress.connect(lambda d, t: self.progress.emit(d, t, title))

        was_idle = not self._active
        self._active[handle] = signals
        self.job_started.emit(title)
        if was_idle:
            self.busy_changed.emit(True)
        self._pool.start(_Job(fn, handle, signals))
        return handle

    def cancel_all(self) -> None:
        for handle in self._active:
            handle.cancel()

    def wait(self, msecs: int = -1) -> bool:
        """Block until all jobs finished (used on shutdown and in tests)."""
        return self._pool.waitForDone(msecs)
