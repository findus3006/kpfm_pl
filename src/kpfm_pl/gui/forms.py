"""Small parameter-form builder used by the tab control panels.

A :class:`ParamForm` is a labelled form whose fields are addressed by a key.
``form.values()`` returns ``{key: value}`` ready to pass to a model, and
``form.set_values`` fills it again. Optional numbers (backend arguments that
accept ``None``) get an "auto" checkbox: unchecked means ``None``.

Example
-------
>>> form = ParamForm()
>>> form.add_float("bintime", "Bin time", 0.05, minimum=1e-4, maximum=60,
...                decimals=4, suffix=" s", tooltip="Width of one time bin.")
>>> form.add_optional_float("tend", "End time", None, maximum=1e6, suffix=" s")
>>> form.values()
{'bintime': 0.05, 'tend': None}
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import Signal as QtSignal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

__all__ = ["ParamForm", "OptionalFloat"]


class OptionalFloat(QWidget):
    """A float spin box with a checkbox; unchecked means "automatic" (``None``)."""

    changed = QtSignal()

    def __init__(self, value: float | None, *, default: float, minimum: float,
                 maximum: float, step: float, decimals: int, suffix: str,
                 auto_text: str = "auto", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.check = QCheckBox()
        self.check.setToolTip(f"Unchecked: {auto_text}")
        self.spin = QDoubleSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setSuffix(suffix)
        self.auto_label = QLabel(auto_text)
        self.auto_label.setStyleSheet("color: gray")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.check)
        layout.addWidget(self.spin, 1)
        layout.addWidget(self.auto_label, 1)
        self._default = default
        self.check.toggled.connect(self._sync)
        self.check.toggled.connect(lambda _: self.changed.emit())
        self.spin.valueChanged.connect(lambda _: self.changed.emit())
        self.set_value(value)

    def _sync(self, enabled: bool) -> None:
        self.spin.setVisible(enabled)
        self.auto_label.setVisible(not enabled)

    def value(self) -> float | None:
        return float(self.spin.value()) if self.check.isChecked() else None

    def set_value(self, value: float | None) -> None:
        self.spin.setValue(self._default if value is None else float(value))
        self.check.setChecked(value is not None)
        self._sync(value is not None)


class ParamForm(QWidget):
    """Labelled form of named parameter fields.

    ``changed`` is emitted whenever any field changes.
    """

    changed = QtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.layout_ = QFormLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.fields: dict[str, QWidget] = {}
        self._labels: dict[str, QLabel] = {}

    # ----------------------------------------------------------------- adding
    def _add(self, key: str, label: str, widget: QWidget, tooltip: str) -> QWidget:
        if key in self.fields:
            raise ValueError(f"Duplicate form key {key!r}")
        label_widget = QLabel(label)
        if tooltip:
            widget.setToolTip(tooltip)
            label_widget.setToolTip(tooltip)
        self.layout_.addRow(label_widget, widget)
        self.fields[key] = widget
        self._labels[key] = label_widget
        return widget

    def add_heading(self, text: str, tooltip: str = "") -> QLabel:
        """Add a bold sub-heading spanning both columns (not a field)."""
        heading = QLabel(f"<b>{text}</b>")
        if tooltip:
            heading.setToolTip(tooltip)
        self.layout_.addRow(heading)
        return heading

    def add_float(self, key: str, label: str, value: float, *, minimum: float = 0.0,
                  maximum: float = 1e9, step: float = 0.1, decimals: int = 3,
                  suffix: str = "", tooltip: str = "") -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setSuffix(suffix)
        spin.setValue(value)
        spin.valueChanged.connect(lambda _: self.changed.emit())
        return self._add(key, label, spin, tooltip)

    def add_optional_float(self, key: str, label: str, value: float | None, *,
                           default: float = 0.0, minimum: float = 0.0,
                           maximum: float = 1e9, step: float = 0.1, decimals: int = 3,
                           suffix: str = "", auto_text: str = "auto",
                           tooltip: str = "") -> OptionalFloat:
        widget = OptionalFloat(value, default=default, minimum=minimum, maximum=maximum,
                               step=step, decimals=decimals, suffix=suffix,
                               auto_text=auto_text)
        widget.changed.connect(self.changed.emit)
        return self._add(key, label, widget, tooltip)

    def add_int(self, key: str, label: str, value: int, *, minimum: int = 0,
                maximum: int = 1_000_000, step: int = 1, suffix: str = "",
                tooltip: str = "") -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setSuffix(suffix)
        spin.setValue(value)
        spin.valueChanged.connect(lambda _: self.changed.emit())
        return self._add(key, label, spin, tooltip)

    def add_bool(self, key: str, label: str, value: bool, *, text: str = "",
                 tooltip: str = "") -> QCheckBox:
        box = QCheckBox(text)
        box.setChecked(value)
        box.toggled.connect(lambda _: self.changed.emit())
        return self._add(key, label, box, tooltip)

    def add_choice(self, key: str, label: str, options: Sequence[tuple[Any, str]],
                   value: Any = None, *, tooltip: str = "") -> QComboBox:
        """Add a drop-down; ``options`` are ``(value, display text)`` pairs."""
        combo = QComboBox()
        for data, text in options:
            combo.addItem(text, data)
        if value is not None:
            index = combo.findData(value)
            if index >= 0:
                combo.setCurrentIndex(index)
        combo.currentIndexChanged.connect(lambda _: self.changed.emit())
        return self._add(key, label, combo, tooltip)

    def set_choices(self, key: str, options: Sequence[tuple[Any, str]]) -> None:
        """Replace the options of a drop-down, keeping the selection if possible."""
        combo = self.fields[key]
        current = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for data, text in options:
            combo.addItem(text, data)
        index = combo.findData(current)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)
        self.changed.emit()

    # ------------------------------------------------------------ reading/writing
    def value(self, key: str) -> Any:
        widget = self.fields[key]
        if isinstance(widget, OptionalFloat):
            return widget.value()
        if isinstance(widget, QDoubleSpinBox):
            return float(widget.value())
        if isinstance(widget, QSpinBox):
            return int(widget.value())
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QComboBox):
            return widget.currentData()
        raise TypeError(f"Unsupported field type for {key!r}")

    def values(self) -> dict[str, Any]:
        return {key: self.value(key) for key in self.fields}

    def set_value(self, key: str, value: Any) -> None:
        widget = self.fields[key]
        if isinstance(widget, OptionalFloat):
            widget.set_value(value)
        elif isinstance(widget, QDoubleSpinBox | QSpinBox):
            widget.setValue(value)
        elif isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QComboBox):
            index = widget.findData(value)
            if index >= 0:
                widget.setCurrentIndex(index)

    def set_values(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            if key in self.fields:
                self.set_value(key, value)

    def set_row_visible(self, key: str, visible: bool) -> None:
        """Show or hide a field together with its label."""
        self.fields[key].setVisible(visible)
        self._labels[key].setVisible(visible)
