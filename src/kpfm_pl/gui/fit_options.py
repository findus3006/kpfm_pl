"""Dialog for the advanced lifetime-fit settings (:class:`FitOptions`).

Initial guesses and bounds per exponential component, the fit statistic
(Poisson MLE by default, or least squares) and a robust loss. The settings are shared by the lifetime
fit, the lifetime trace and the FLIDs of the *PL lifetime* tab. The BIC
threshold and the over-fitting warning thresholds only apply to the lifetime
fit.
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
from .models.ptu import (
    BIC_THRESHOLD,
    BIN_DEFAULT_WEIGHTING,
    DIAGNOSTIC_DEFAULTS,
    FIT_DEFAULT_WEIGHTING,
    LOSSES,
    MAX_COMPONENTS,
    MEAN_LIFETIMES,
    WEIGHTINGS,
    FitOptions,
)

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


#: FitOptions fractions that the dialog shows in percent.
_PERCENT_FIELDS = ("min_fraction", "merge_rel", "bound_rel", "period_fraction")


def describe_options(options: FitOptions) -> str:
    """One-line summary of the non-default settings (``"defaults"`` if none)."""
    parts = []
    for attr, title, _spin, _tip in _COLUMNS:
        used = [i + 1 for i, v in enumerate(getattr(options, attr)) if v is not None]
        if used:
            parts.append(f"{title} ({', '.join(map(str, used))})")
    names = {"mle": "Poisson MLE", "poisson": "Poisson least squares",
             "none": "plain least squares"}
    if options.weighting != FIT_DEFAULT_WEIGHTING:
        parts.append(f"lifetime fit: {names[options.weighting]}")
    if options.bin_weighting != BIN_DEFAULT_WEIGHTING:
        parts.append(f"trace/FLID: {names[options.bin_weighting]}")
    if options.loss != "linear":
        parts.append(f"{options.loss} loss (scale {options.f_scale:g})")
        if "mle" in (options.weighting, options.bin_weighting):
            parts.append("MLE replaced by Poisson least squares")
    if options.constrain_sum_A:
        parts.append("sum(A) ≤ 1 (per-bin fits)")
    if options.mean_lifetime != "intensity":
        parts.append("amplitude-weighted mean lifetime")
    if not options.decay_correction:
        parts.append("no incomplete-decay correction")
    if options.bic_threshold != BIC_THRESHOLD:
        parts.append(f"|ΔBIC| threshold {options.bic_threshold:g}")
    if any(getattr(options, k) != v for k, v in DIAGNOSTIC_DEFAULTS.items()):
        parts.append("custom warning thresholds")
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
        f.add_heading("Fit statistic",
                      tooltip="How the deviation between data and model is measured. "
                              "Separate for the lifetime fit and for the many small "
                              "per-bin fits of the lifetime trace and the FLIDs.")
        f.add_choice("weighting", "Lifetime fit", WEIGHTINGS, FIT_DEFAULT_WEIGHTING,
                     tooltip="Poisson MLE (default): maximum-likelihood fit of the photon "
                             "counts. Correct also where the histogram holds few photons "
                             "per bin (long tails), where least squares underestimates "
                             "the lifetimes. The BIC then uses the Poisson deviance "
                             "instead of χ². If the MLE does not converge (e.g. a fixed "
                             "background far too low), the Poisson least-squares fit is "
                             "shown instead and the log says so.\n"
                             "Poisson least squares: bins are weighted by their photon "
                             "counting noise 1/√counts.\n"
                             "None: every histogram bin counts equally (plain least "
                             "squares), so the low-count tail is nearly ignored.")
        f.add_choice("bin_weighting", "Lifetime trace / FLID", WEIGHTINGS,
                     BIN_DEFAULT_WEIGHTING,
                     tooltip="The per-bin fits of the lifetime trace and the FLIDs.\n"
                             "Poisson least squares (default): fast and close to the "
                             "maximum likelihood from about 100 photons per time bin.\n"
                             "Poisson MLE: unbiased also with few photons per bin, but "
                             "several times slower. Bins where it does not converge get "
                             "the Poisson least-squares value; the log gives their "
                             "number.\n"
                             "None: plain least squares, biased towards short lifetimes "
                             "at low counts.")
        f.add_heading("Loss and constraint")
        f.add_choice("loss", "Loss", LOSSES, "linear",
                     tooltip="Linear: the standard fit. The others are robust losses that "
                             "reduce the influence of outliers (e.g. afterpulsing "
                             "spikes). They are a least-squares concept: with Poisson MLE "
                             "selected, the fits then use Poisson least squares and the "
                             "log says so.")
        f.add_float("f_scale", "Loss scale", 1.0, minimum=1e-4, maximum=1e4, step=0.1,
                    decimals=4,
                    tooltip="Residual size where a robust loss starts to down-weight "
                            "points: in Poisson standard deviations, or as a fraction of "
                            "the histogram peak with weighting 'None'.")
        f.add_bool("constrain_sum_A", "Constraint", False, text="sum(A) ≤ 1",
                   tooltip="Keep the amplitudes of all components below one peak in total, "
                           "so they stay fractions of the decay. Applied only to the "
                           "normalised per-bin fits of the lifetime trace and the FLIDs. "
                           "The lifetime fit works on raw counts and ignores it (its "
                           "amplitudes are counts; see the % rows of the results table "
                           "for fractions).")
        f.add_heading("Amplitudes and mean lifetime")
        f.add_choice("mean_lifetime", "Mean lifetime", MEAN_LIFETIMES, "intensity",
                     tooltip="How several exponentials are averaged into one lifetime, in "
                             "the fit table and title, the lifetime trace and the FLID "
                             "maps.\nIntensity-weighted τ_int = Σaτ²/Σaτ: weighted by "
                             "photons, equals the mean photon arrival time (comparable "
                             "with the 'Mean arrival time' FLID).\nAmplitude-weighted "
                             "τ_amp = Σaτ/Σa: weighted by the share of emitters, e.g. for "
                             "quenching or FRET.")
        f.add_bool("decay_correction", "Incomplete decay", True,
                   text="correct amplitudes for earlier pulses",
                   tooltip="If the decay has not reached the noise floor within one laser "
                           "period, every bin also holds photons of earlier pulses and "
                           "the fitted amplitudes are too large (long components most): "
                           "A = A_fit·(1 − exp(−T/τ)). The lifetimes are not affected. "
                           "Applies to the lifetime fit, the lifetime trace and the FLIDs.\n"
                           "Untick to report the fitted amplitudes unchanged.")
        f.add_heading("Model selection (lifetime fit only)",
                      tooltip="How a lifetime fit is compared with the previous one and "
                              "when it warns about too many components.")
        f.add_float("bic_threshold", "|ΔBIC| threshold", 10.0, minimum=0.0, maximum=1e4,
                    step=1.0, decimals=1,
                    tooltip="BIC = χ² + k·ln n (Poisson deviance instead of χ² for the "
                            "MLE), k free parameters, n fitted bins. Each fit is compared "
                            "with the previous fit of the same histogram, window and "
                            "weighting: ΔBIC below −threshold is shown green (better), "
                            "above +threshold red (worse, e.g. an extra component that "
                            "only fits noise), otherwise gray.\nKass & Raftery (1995): "
                            "2 positive, 6 strong, 10 very strong evidence.")
        f.add_float("amp_sigma", "Amplitude ≈ 0 below", 2.0, minimum=0.0, maximum=100.0,
                    step=0.5, decimals=1, suffix=" σ",
                    tooltip="Warn when an amplitude is smaller than this many standard "
                            "errors, i.e. compatible with zero.")
        f.add_float("min_fraction", "Collapsed below", 1.0, minimum=0.0, maximum=100.0,
                    step=0.5, decimals=2, suffix=" % intensity",
                    tooltip="Warn when a component carries less than this fraction of "
                            "the intensity Aᵢτᵢ / Σ Aⱼτⱼ.")
        f.add_float("merge_sigma", "Lifetimes merged within", 2.0, minimum=0.0,
                    maximum=100.0, step=0.5, decimals=1, suffix=" σ",
                    tooltip="Warn when two lifetimes differ by less than this many "
                            "combined standard errors …")
        f.add_float("merge_rel", "… or within", 10.0, minimum=0.0, maximum=100.0, step=1.0,
                    decimals=1, suffix=" %",
                    tooltip="… or by less than this percentage of the larger lifetime.")
        f.add_float("bound_rel", "At a bound within", 1.0, minimum=0.0, maximum=100.0,
                    step=0.5, decimals=2, suffix=" %",
                    tooltip="Warn when a fitted τ or A ends within this percentage of one "
                            "of its bounds (Advanced bounds above): the fit wants to go "
                            "beyond it.")
        f.add_float("period_fraction", "τ vs. background above", 50.0, minimum=0.0,
                    maximum=1000.0, step=5.0, decimals=0, suffix=" % of the period",
                    tooltip="Only with a fitted background: warn when a lifetime exceeds "
                            "this percentage of the laser period. Such a slow decay is "
                            "nearly flat within one period (at 50 % it only falls to "
                            "14 %), so it and the constant background are barely "
                            "separable and both become uncertain.")
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
        values = {"weighting": options.weighting, "bin_weighting": options.bin_weighting,
                  "loss": options.loss,
                  "f_scale": options.f_scale, "constrain_sum_A": options.constrain_sum_A,
                  "mean_lifetime": options.mean_lifetime,
                  "decay_correction": options.decay_correction,
                  "bic_threshold": options.bic_threshold, "amp_sigma": options.amp_sigma,
                  "merge_sigma": options.merge_sigma}
        # Fractions are entered in percent.
        values.update({key: 100.0 * getattr(options, key) for key in _PERCENT_FIELDS})
        self.form.set_values(values)
        self._update_fields()

    def options(self) -> FitOptions:
        values = {attr: [self.cells[(attr, i)].value() for i in range(MAX_COMPONENTS)]
                  for attr, *_ in _COLUMNS}
        form = self.form.values()
        form.update({key: form[key] / 100.0 for key in _PERCENT_FIELDS})
        return FitOptions(**values, **form)

    def _accept(self) -> None:
        try:
            self.options().validate()
        except ValueError as exc:
            self.error_label.setText(str(exc))
            return
        self.accept()
