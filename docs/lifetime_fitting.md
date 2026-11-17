# Fitting fluorescence lifetimes – the method

This note explains how `kpfm_pl` turns a TCSPC lifetime histogram into
lifetimes, amplitudes and mean lifetimes, and what to watch when judging the
result. It is written for experimentalists. How to run the fits is in
[ptu.md](ptu.md#measuring-a-lifetime) (Python) and [gui.md](gui.md#pl-lifetime-ptu)
(GUI); the fit with a measured instrument response is explained in
[irf_reconvolution.md](irf_reconvolution.md). Bracketed numbers refer to the
references at the end.

The example numbers come from the sample file `ptu_test1` (110 591 photons,
laser period 102.9 ns, 64 ps bins, dark counts 2.25 /s).

## 1. What the histogram contains

In time-correlated single photon counting every detected photon is sorted by
its delay after the laser pulse that excited it. Summed over many pulses
this gives the lifetime histogram. Four things shape it:

- **The decay** of the sample, for one species `A exp(−t/τ)`, for several a
  sum of such terms.
- **The instrument response** (IRF): laser pulse width, timing jitter and the
  transit-time spread of the detector blur every delay by typically
  50–500 ps. The rising edge and the top of the peak are dominated by it.
- **Periodic excitation**: the laser fires every `T` (about 100 ns at
  10 MHz). A decay that has not died away within one period is still emitting
  when the next pulse arrives, so every bin also contains photons of earlier
  pulses.
- **A flat background**: detector dark counts, stray light, afterpulsing.

Photon counting obeys Poisson statistics: a bin expected to hold `m` counts
fluctuates by `√m`. This decides how the fit has to weight the bins
(section 5).

## 2. The tail fit

Without a recorded IRF, `kpfm_pl` fits the decay after its maximum, where the
instrument response has (nearly) died away [1, 7]:

```
y(t) = Σᵢ Aᵢ exp(−t/τᵢ) + C          (one to four components)
```

`t = 0` is placed at the histogram maximum, so the fitted `Aᵢ` are the
amplitudes of the decay extrapolated to the maximum, not to the excitation
time.

**Fit window.** The window starts after the maximum so that the rising edge
and the IRF-shaped top are left out:

- *Falling edge (fraction)*: where the smoothed decay has fallen to a given
  fraction (default 95 %) of the maximum;
- *Inflection point*: at the steepest point of the falling edge;
- *Manual*: between a chosen start and end.

A later start removes more of the IRF influence on short components but also
removes photons. A tail fit is reliable when every lifetime is much longer
than the IRF width (roughly 5–10 times). For shorter lifetimes, or when the
amplitudes themselves matter, use the IRF reconvolution
([irf_reconvolution.md](irf_reconvolution.md)).

## 3. Periodic excitation

### Incomplete decay

With pulses every `T`, the contributions of all earlier pulses add up. For one
exponential the sum has a closed form:

```
Σₙ exp(−(t + nT)/τ) = exp(−t/τ) / (1 − exp(−T/τ))
```

The shape is still a single exponential with the same `τ`: **the fitted
lifetimes are correct**. The fitted amplitude, however, is too large by
`1 / (1 − exp(−T/τ))`, most for the long components:

| `τ` at `T = 100 ns` | 5 ns | 20 ns | 50 ns |
|---|---|---|---|
| amplitude too large by | 0.0 % | 0.7 % | 16 % |

Uncorrected, this pushes the intensity-weighted mean lifetime and the
intensity fractions towards the long components. `kpfm_pl` therefore reports
the **per-pulse amplitudes** `A = A_fit · (1 − exp(−T/τ))` and computes all
mean lifetimes and fractions from them; the fitted `A_fit` describe the drawn
curve. The correction can be switched off (*Incomplete decay*). On
`ptu_test1` (2 components, 6.9 and 31.9 ns) it lowers the long amplitude from
109.3 to 105.0 counts and `τ_int` from 19.75 to 19.49 ns.

### The bins before the rise

With an incomplete decay the bins before the rising edge are not empty: they
hold the tail of the previous pulse, one period later than their position
suggests. They can be fitted as well, at the time `t + T` (*Before the rise:
also fit these bins*). This extends the observed decay from the fit window
to almost the whole period, which helps for decays that have not reached the
background within one period, e.g. long-lived nanocrystals. A gap before the
maximum (default 2 ns) keeps the rising edge itself out of the fit.

## 4. The background

The constant `C` can be fixed or fitted:

- **Fixed at the dark counts** (default): `C` = dark-count rate × measuring
  time / number of lifetime bins. Correct when nothing but dark counts lies
  under the decay.
- **Fitted** (*fit background B*): needed when the histogram sits on a
  pedestal that the dark counts do not explain: stray light, afterpulsing,
  or an emission much slower than the period. A fixed offset that is too
  low is not harmless: the fit then describes the pedestal with its longest
  lifetime.

On `ptu_test1` the two choices tell different stories. With the dark counts
fixed (0.50 counts per bin) two exponentials leave systematic residuals
(χ²ᵣ = 1.41) and three give 1.8 / 10.7 / 53 ns. With a fitted background the
fit finds 10.8 ± 0.3 counts per bin and two exponentials, 2.9 / 14.3 ns,
with χ²ᵣ = 1.19. Which picture is right is a question about the sample and
the setup, not about the fit: measure the background independently (e.g.
the count rate while the laser is blocked) if the two differ.

**Long lifetimes and the background are hard to separate.** Within one
period a slow decay is almost flat: at `τ = T/2` it only falls to 14 %.
With a fitted background `kpfm_pl` therefore warns about every lifetime above
half the laser period. Then fix the background at a measured level or record
with a lower repetition rate. The bins before the rise (section 3) extend the
time range and reduce, but do not remove, the ambiguity.

## 5. Counting statistics: how the bins are weighted

The fit minimises a measure of the distance between data and model. Three are
available:

| Statistic | Minimises | Behaviour |
|---|---|---|
| **Poisson maximum likelihood** (MLE) | the Poisson deviance `2 Σ [m − y + y ln(y/m)]` | the correct statistic for photon counts; unbiased also where bins hold few photons |
| **Poisson least squares** (Neyman χ²) | `Σ (y − m)² / max(y, 1)` | weights every bin by its counting noise; biased towards short lifetimes where bins hold few counts, because bins that fluctuate low get more weight |
| **Plain least squares** | `Σ (y − m)²` | every bin counts equally, so the well-filled peak dominates and the low-count tail is nearly ignored |

With many counts in every fitted bin the first two agree. They differ in the
long tail of a histogram and in the per-bin fits of a lifetime trace or FLID.
Maus et al. found that least squares underestimates single-molecule lifetimes
by about 5 % at 1 000–20 000 photons, while the MLE stays accurate [15]; a
simulation with the decay of `ptu_test1` gives 28.6 ns for the long
component with Poisson least squares and 30.3 ns with the MLE (true:
30.2 ns). On the real file (2 components) plain least squares gives
3.4 / 18.6 ns, Poisson least squares 6.8 / 29.3 ns and the MLE 6.9 / 31.9 ns.

**Defaults.** The lifetime fit of a summed histogram uses the MLE. The many
small fits of a lifetime trace or FLID use Poisson least squares: from
about 100 photons per time bin it gives nearly the same lifetimes and is
several times faster. The MLE can be chosen there as well.

The MLE is computed iteratively from the Poisson least-squares result. If it
does not converge, typically because the model does not describe the data
(e.g. a fixed background far below the real one), the Poisson least-squares
result is shown and marked as such. A robust loss (against outliers such as
afterpulsing spikes) exists only for least squares; with it the MLE is
replaced by Poisson least squares as well.

Always fit **raw counts**: scaling a histogram destroys the relation between
a count and its noise.

## 6. Is the fit good?

- **Reduced χ²** (computed with Poisson weights on the raw counts): about 1
  when the model describes the data within photon noise; clearly above 1
  means systematic deviations (too few components, a badly placed window, a
  wrong background); below 1 can mean over-fitting or very few photons.
- **Weighted residuals** `(y − m)/√y` against time: they should scatter
  around zero without structure. A wave or a trend shows where the model
  fails.
- **R²** is close to 1 for almost any decay and says little.

## 7. How many components?

Adding an exponential always lowers χ², even if the new component only fits
noise. The **Bayesian information criterion** [21, 22] adds a penalty per free
parameter:

```
BIC = χ² + k ln n        (with the MLE: deviance + k ln n)
```

with `k` free parameters (two per component, one more for a fitted
background, and the IRF parameters with reconvolution) and `n` fitted bins.
The lower BIC wins. Compare only fits of the **same histogram, fit window
and statistic**; otherwise the numbers are not comparable. Kass & Raftery
grade a difference of 2–6 as positive, 6–10 as strong and above 10 as very
strong evidence [22]. In the GUI each fit is compared with the previous one
(green: better by more than the threshold, default 10; red: worse). Fit one,
then two, then three components and keep the last clear improvement.

Signs that a component is too many, reported as warnings:

- an amplitude compatible with zero (below two standard errors) or carrying
  less than 1 % of the photons;
- two lifetimes that have merged (closer than two combined errors or 10 %);
- a parameter at one of its bounds.

With very many photons the BIC keeps rewarding extra components for any
small imperfection (IRF influence, a slightly wrong background). Accept an
extra component only if it is also plausible physically.

## 8. Mean lifetimes and fractions

From the per-pulse amplitudes `Aᵢ`:

| Quantity | Formula | Meaning |
|---|---|---|
| intensity fraction `fᵢ` | `Aᵢτᵢ / Σ Aⱼτⱼ` | share of the emitted photons in component `i` |
| amplitude fraction | `Aᵢ / Σ Aⱼ` | share of the excited emitters in component `i` |
| `τ_int` | `Σ Aᵢτᵢ² / Σ Aᵢτᵢ` | intensity-weighted mean, equal to the mean photon arrival time; the usual "average lifetime" of FLIM and single-emitter work |
| `τ_amp` | `Σ Aᵢτᵢ / Σ Aᵢ` | amplitude-weighted mean, proportional to the quantum yield in quenching and FRET analyses [1] |

On `ptu_test1` (2 components) `τ_int` = 19.5 ns but `τ_amp` = 11.4 ns:
choose the one that answers your question and say which. Because the tail
fit puts `t = 0` at the maximum, the amplitudes refer to that time, which
weights the short components slightly less than at the excitation time; the
IRF reconvolution has no such dependence.

**Uncertainties.** The parameter errors come from the curvature of the fit
(the covariance matrix), scaled by the reduced χ², so a poor fit also gets
larger errors. Lifetimes and amplitudes of a multi-exponential fit are
strongly correlated; the errors of `τ_int`, `τ_amp` and the fractions are
propagated including these correlations. That is why `τ_int` is usually
better defined than the individual `τᵢ`: on `ptu_test1` `τ_int` = 19.49 ±
0.20 ns (1 %), while the two lifetimes are 6.92 ± 0.13 ns and 31.9 ± 0.7 ns
(2 %).

## 9. Seam bins and the lifetime shift

The laser period is usually not a whole number of lifetime bins (102.94 ns /
64 ps = 1608.4 bins). The last bin of the period is then only partly covered
and collects fewer photons, and bins past the period never fill. Such
**seam bins** do not follow the decay and are left out of every fit by
default.

A recording with a wrong sync offset has its decay late in the window, with
the tail continuing at the window start. The **lifetime shift** rotates the
delays by whole bins so that the decay is in one piece again; the seam bins
then sit inside the decay and are still left out. Because the rotation is in
whole bins while the period is not, the bins moved across the window edge
end up a fraction of a bin too late (0.30 bin = 19 ps on `ptu_test2`; see
[ptu.md](ptu.md#lifetime-shift-salvaging-a-wrong-sync-offset)). The effect on
the lifetimes is about 0.01 %, far below their uncertainty. The bins before
the rise (section 3) use the exact period and have no such rounding.

## 10. Lifetimes per time bin: lifetime trace and FLID

For blinking or otherwise changing emitters, the photons of each short time
bin are fitted separately (lifetime trace), and the pairs of lifetime and
count rate are collected in a two-dimensional histogram (FLID,
fluorescence-lifetime–intensity distribution).

- **Photons per bin.** A bin with `N` photons gives a lifetime that scatters
  by roughly `τ/√N` from photon noise alone [13]. About 100 photons per bin
  are a practical minimum for one exponential; `kpfm_pl` checks this and
  suggests a longer bin time. Two components per bin need far more photons.
- **Shared parameters.** A bin holds too few photons to determine its own
  background or IRF shift. These are taken from the fit of the summed
  histogram (background, IRF shift and scatter of the last lifetime fit),
  and only amplitudes and lifetimes are fitted per bin.
- **Smoothing of the FLID** adds its own width to the photon-noise spread.
  Judge the width of a FLID cloud only without smoothing.
- **Mean arrival time** as a fit-free cross-check: the mean delay of the
  photons in the fit window equals `τ` for one exponential. It is biased by
  the finite window (corrected) and by the background (dark counts are
  subtracted first). For several components it gives the `τ_int` of the
  decay at the window start, which is longer than the fitted `τ_int`. A
  clear difference between the fitted and the mean-arrival FLID shows that
  one exponential per bin is not enough.

## 11. A practical sequence

1. Check the histogram: is the peak early in the window, is the period
   covered, is there a pedestal?
2. Fit one, two and three components with the MLE; keep the number the BIC
   supports and the warnings allow.
3. Look at the residuals. If they show a pedestal or a slow tail, compare a
   fixed and a fitted background and decide with an independent background
   measurement.
4. For long-lived samples include the bins before the rise.
5. If lifetimes approach the IRF width, or amplitudes matter, record an IRF
   and use the reconvolution.
6. Report the mean lifetime you use (`τ_int` or `τ_amp`) with its error, the
   number of components, the fit window, the statistic and the background
   treatment.

## References

Numbering as in [irf_reconvolution.md](irf_reconvolution.md#references),
where the full list is given.

1. J. R. Lakowicz, *Principles of Fluorescence Spectroscopy*, 3rd ed.,
   Springer, 2006, ch. 4.
7. PicoQuant / tcspc.com knowledge base: "Lifetime fitting using the TCSPC
   fitting script".
13. M. Köllner, J. Wolfrum, "How many photons are necessary for
    fluorescence-lifetime measurements?", *Chem. Phys. Lett.* **200**, 199
    (1992).
15. M. Maus et al., "An experimental comparison of the maximum likelihood
    estimation and nonlinear least-squares fluorescence lifetime analysis of
    single molecules", *Anal. Chem.* **73**, 2078 (2001).
21. G. Schwarz, "Estimating the dimension of a model", *Ann. Statist.* **6**,
    461 (1978).
22. R. E. Kass, A. E. Raftery, "Bayes factors", *J. Am. Stat. Assoc.* **90**,
    773 (1995).
