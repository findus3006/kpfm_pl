"""Common layout and helpers for the workflow tabs.

Every tab has a scrollable control column on the left (an explanatory hint
followed by group boxes of parameters and buttons) and a content area on the
right, usually one or more :class:`~kpfm_pl.gui.canvas.PlotPanel`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractButton,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..workers import JobHandle

__all__ = ["BaseTab", "CollapsibleGroup"]


class CollapsibleGroup(QWidget):
    """A group box whose title button shows or hides its content."""

    def __init__(self, title: str, *, collapsed: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.button = QToolButton()
        self.button.setText(title)
        self.button.setCheckable(True)
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.button.setToolTip("Click to show or hide this group.")
        self.body = QGroupBox()
        self.content_layout = QVBoxLayout(self.body)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.button)
        layout.addWidget(self.body)
        self.button.toggled.connect(self.set_expanded)
        self.button.setChecked(not collapsed)
        self.set_expanded(not collapsed)

    def set_expanded(self, expanded: bool) -> None:
        if self.button.isChecked() != expanded:
            self.button.setChecked(expanded)
        self.button.setArrowType(Qt.ArrowType.DownArrow if expanded
                                 else Qt.ArrowType.RightArrow)
        self.body.setVisible(expanded)

    def is_expanded(self) -> bool:
        return self.button.isChecked()


class BaseTab(QWidget):
    """Base class for the main-window tabs.

    Subclasses set :attr:`title`, build their controls with :meth:`add_hint`,
    :meth:`add_group` and :meth:`add_button`, put their plots into
    :attr:`content_layout` and override :meth:`on_file_changed` to react to
    newly loaded files.
    """

    title = "Tab"

    def __init__(self, ctx, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.groups: dict[str, QWidget] = {}

        controls = QWidget()
        self.controls_layout = QVBoxLayout(controls)
        self.controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll = QScrollArea()
        scroll.setWidget(controls)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(320)

        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(content)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1000])
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        ctx.state.file_changed.connect(self.on_file_changed)

    # ---------------------------------------------------------------- building
    def add_hint(self, text: str) -> QLabel:
        """Add a short grey explanation to the control column."""
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: palette(placeholder-text);")
        self.controls_layout.addWidget(label)
        return label

    def add_group(self, title: str, *, collapsible: bool = False,
                  collapsed: bool = False) -> QVBoxLayout:
        """Add a titled group box to the control column and return its layout.

        A ``collapsible`` group can be folded away by clicking its title; it
        starts folded when ``collapsed``. The group widget is kept in
        :attr:`groups` under its title.
        """
        if collapsible:
            group = CollapsibleGroup(title, collapsed=collapsed)
            layout = group.content_layout
        else:
            group = QGroupBox(title)
            layout = QVBoxLayout(group)
        self.groups[title] = group
        self.controls_layout.addWidget(group)
        return layout

    @staticmethod
    def add_button(layout: QVBoxLayout, text: str, slot: Callable[[], Any], *,
                   tooltip: str = "", primary: bool = False) -> QPushButton:
        button = QPushButton(text)
        if tooltip:
            button.setToolTip(tooltip)
        if primary:
            button.setDefault(True)
            font = button.font()
            font.setBold(True)
            button.setFont(font)
        button.clicked.connect(lambda _=False: slot())
        layout.addWidget(button)
        return button

    # ------------------------------------------------------------------ jobs
    def run_job(self, title: str, fn: Callable[[Callable[[int, int], None]], Any],
                on_done: Callable[[Any], None] | None = None, *,
                disable: Iterable[QAbstractButton] = (),
                cancellable: bool = False,
                on_error: Callable[[str, str], None] | None = None) -> JobHandle:
        """Run ``fn(progress)`` in the background with ``disable`` greyed out.

        Errors are shown by the main window (message box + log); pass
        ``on_error`` for additional, tab-specific handling.
        """
        widgets = list(disable)
        for widget in widgets:
            widget.setEnabled(False)

        def _restore() -> None:
            for widget in widgets:
                widget.setEnabled(True)

        return self.ctx.runner.submit(title, fn, on_done, on_error=on_error,
                                      on_finally=_restore, cancellable=cancellable)

    # ---------------------------------------------------------------- hooks
    def on_file_changed(self, kind: str, handle) -> None:
        """Called when a file of ``kind`` was loaded; override in subclasses."""
