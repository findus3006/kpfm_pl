"""Dialog for the advanced lifetime-fit settings (:class:`FitOptions`).

Initial guesses and bounds per exponential component, Poisson weighting and
a robust loss. The settings are shared by the lifetime fit, the lifetime
trace and the FLIDs of the *PL lifetime* tab.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .forms import OptionalFloat, ParamForm
from .models.ptu import LOSSES, MAX_COMPONENTS, WEIGHTINGS, FitOptions

__all__ = ["FitOptionsDialog", "describe_options"]

#: (FitOptions attribute, column title, spin-box settings, tooltip)
_COLUMNS = (
    ("tau_guess", "τ guess", dict(maximum=1e4, step=0.5, decimals=3, suffix=" ns"),
     "Initial guess of this lifetime. Unchecked: automatic."),
    ("tau_min", "τ min", dict(maximum=1e4, step=0.5, decimals=3, suffix=" ns"),
     "Lower bound of this lifetime. Unchecked: > 0."),
    ("tau_max", "τ max", dict(maximum=1e4, step=0.5, decimals=3, suffix=" ns"),
     "Upper bound of this lifetime. Unchecked: no limit."),
    ("amp_guess", "A guess", dict(maximum=100, step=0.05, decimals=3),
     "Initial guess of this amplitude, as a fraction of the histogram peak. "
     "Unchecked: automatic."),
    ("amp_min", "A min", dict(minimum=-100, maximum=100, step=0.05, decimals=3),
     "Lower bound of this amplitude (fraction of the histogram peak). Unchecked: 0, "
     "i.e. no rise components."),
    ("amp_max", "A max", dict(maximum=100, step=0.05, decimals=3),
     "Upper bound of this amplitude (fraction of the histogram peak). Unchecked: no "
     "limit."),
)


def describe_options(options: FitOptions) -> str:
    """One-line summary of the non-default settings (``"defaults"`` if none)."""
    parts = []
    for attr, title, _spin, _tip in _COLUMNS:
        used = [i + 1 for i, v in enumerate(getattr(options, attr)) if v is not None]
        if used:
            parts.append(f"{title} ({', '.join(map(str, used))})")
    if options.weighting != "none":
        parts.append(f"{options.weighting} weighting")
    if options.loss != "linear":
        parts.append(f"{options.loss} loss (scale {options.f_scale:g})")
    return ", ".join(parts) if parts else "defaults"


class FitOptionsDialog(QDialog):
    """Edit a :class:`FitOptions`; :meth:`options` returns the entered values."""

    def __init__(self, options: FitOptions | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Advanced fit settings")
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Used by the lifetime fit, the lifetime trace and the FLIDs. Tick a field "
            "to set it; unticked fields are automatic. Components beyond the number "
            "of exponentials of a fit are ignored. Amplitudes are fractions of the "
            "histogram peak (the per-bin fits are normalised to 1)."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        grid = QGridLayout()
        self.cells: dict[tuple[str, int], OptionalFloat] = {}
        for col, (attr, title, spin, tip) in enumerate(_COLUMNS, 1):
            header = QLabel(f"<b>{title}</b>")
            header.setToolTip(tip)
            grid.addWidget(header, 0, col)
            for i in range(MAX_COMPONENTS):
                settings = {"minimum": 0.0, "suffix": "", **spin}
                cell = OptionalFloat(None, default=1.0, auto_text="auto", **settings)
                cell.setToolTip(tip)
                grid.addWidget(cell, i + 1, col)
                self.cells[(attr, i)] = cell
        for i in range(MAX_COMPONENTS):
            grid.addWidget(QLabel(f"Component {i + 1}"), i + 1, 0)
        layout.addLayout(grid)

        f = self.form = ParamForm()
        f.add_choice("weighting", "Weighting", WEIGHTINGS, "none",
                     tooltip="None: every histogram bin counts equally (standard least "
                             "squares).\nPoisson: bins are weighted by their photon "
                             "counting noise 1/√counts, so the low-count tail is not "
                             "ignored. Usually the better choice for TCSPC data.")
        f.add_choice("loss", "Loss", LOSSES, "linear",
                     tooltip="Linear: standard least squares. The others are robust "
                             "losses that reduce the influence of outliers (e.g. "
                             "afterpulsing spikes).")
        f.add_float("f_scale", "Loss scale", 1.0, minimum=1e-4, maximum=1e4, step=0.1,
                    decimals=4,
                    tooltip="Residual size where a robust loss starts to down-weight "
                            "points: in Poisson standard deviations with Poisson "
                            "weighting, otherwise as a fraction of the histogram peak.")
        f.changed.connect(self._update_fields)
        layout.addWidget(f)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        reset = QPushButton("Reset to defaults")
        reset.setToolTip("Untick every field and use the standard fit.")
        reset.clicked.connect(lambda: self.set_options(FitOptions()))
        buttons.addButton(reset, QDialogButtonBox.ButtonRole.ResetRole)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #c00000")
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        layout.addWidget(buttons)

        self.set_options(options or FitOptions())

    def _update_fields(self) -> None:
        self.form.set_row_visible("f_scale", self.form.value("loss") != "linear")

    def set_options(self, options: FitOptions) -> None:
        for (attr, i), cell in self.cells.items():
            cell.set_value(getattr(options, attr)[i])
        self.form.set_values({"weighting": options.weighting, "loss": options.loss,
                              "f_scale": options.f_scale})
        self._update_fields()

    def options(self) -> FitOptions:
        values = {attr: [self.cells[(attr, i)].value() for i in range(MAX_COMPONENTS)]
                  for attr, *_ in _COLUMNS}
        return FitOptions(**values, **self.form.values())

    def _accept(self) -> None:
        try:
            self.options().validate()
        except ValueError as exc:
            self.error_label.setText(str(exc))
            return
        self.accept()
