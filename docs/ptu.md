# PTU reader (`kpfm_pl.PTU`)

Loads PicoQuant `.ptu` TTTR-T3 files (TCSPC / photoluminescence) and exposes
them as binned **time traces**, **lifetime histograms**, **FLID maps** and
**multi-exponential lifetime fits**. Primary plottable data products use the
[`Signal`](../src/kpfm_pl/core.py) or [`Image`](../src/kpfm_pl/core.py)
containers; fit summaries, triggers and raw event access use their documented
dictionary/array forms. There is no plotting inside the loader.

The runnable example is
[`examples/example_ptu.py`](../examples/example_ptu.py), which uses
`ptu_test2_LaserPulse(...).ptu` from the `Test data/` folder.

## Loading a file

```python
from kpfm_pl import PTU

ptu = PTU("scan.ptu", dark_counts=2.25)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `filepath` | – | path to the `.ptu` file |
| `dark_counts` | `0.0` | dark counts per second, subtracted as a constant offset `C` during lifetime fitting |
| `smooth_rep_rate` | `False` | use the rounded laser repetition time for record decoding |
| `lifetime_shift` | `None` | circular shift of the lifetime axis, see [Lifetime shift](#lifetime-shift-salvaging-a-wrong-sync-offset) |

On load the header tags and all photon records are parsed. Continuous-wave
files (no laser trigger / infinite global resolution) are detected and the
photon records are skipped with a printed notice; the `is_cw` attribute is set
to `True`, the lifetime axis stays empty (`n_lt_bins = 0`) and every
time-binned/lifetime method (`time_trace`, `lifetime_histogram`, `flid`,
`threshold`, `fit_lifetime`, `laser_triggers`, `get_markers`) raises a clear
`ValueError` instead of failing on a half-constructed object.

Useful attributes after loading:

| Attribute | Meaning |
|-----------|---------|
| `acquisition_time` | total measurement time (s) |
| `resolution` | TCSPC bin resolution (s) |
| `rep_time` | laser repetition period (s; `inf` for continuous-wave files) |
| `is_cw` | `True` for continuous-wave files (see above) |
| `n_lt_bins`, `lt_time` | lifetime-axis length and time axis (s) |
| `events` | raw photon array, shape `(N, 3)` = `[truesync, dtime, time_s]` |
| `marker` | `{channel: [sync, ...]}` hardware markers |
| `tags` | all header tags as a `dict` |

## Time trace

```python
tt = ptu.time_trace(bintime=0.05, laser_pulse=True)   # Signal (x=s, y=counts)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `bintime` | `0.05` | width of each time bin (s) |
| `tstart`, `tend` | `0.0`, `None` | optional time window; `None` = full acquisition |
| `laser_pulse` | `True` | see below |

The requested time window is interpreted as the half-open interval
`[tstart, tend)`. Every acquisition bin that overlaps it is retained, including
a partial final bin, and `Signal.x` contains the selected bins' actual left-edge
times. Thus an unaligned `tstart` can return a first x value earlier than
`tstart` when that bin overlaps the requested interval.

The `laser_pulse` flag refers to the **pulse generator in the laser driver**
(e.g. a 10 MHz pulsed laser sends a pulse every 100 ns, which the PicoHarp
needs as its sync reference):

- `laser_pulse=True` (**default**, and how data is recorded from now on) – the
  pulse generator emits triggers throughout the whole file, regardless of
  whether the laser light is on. Those triggers are not recorded in the marker
  channel and are not to be confused with the laser triggers from `laser_trigger()`
  and `get_markers()`.
- `laser_pulse=False` – a **legacy work-around** for files where no marker was
  picked up (e.g. the pulse generator only ran while the laser was on). The
  non-zero portion of the trace is centred and zero-padded, and the
  rising/falling edges are stored as triggers in `tt["meta"]["triggers"]`.
  This is bad practice and only kept for backwards compatibility.

The returned `Signal.meta` carries `bintime`, the per-bin `matrix`
(time-bin × lifetime-bin counts) and `average_counts`. The row sum of `matrix`
always equals `Signal.y`.

## Lifetime histogram

```python
hist = ptu.lifetime_histogram()          # sums the cached time-trace matrix
hist = ptu.lifetime_histogram(matrix, bintime=0.05)  # explicit count matrix
```

Returns a `Signal` of counts vs. lifetime-bin time (s). If no time trace has
been built yet, a default one is created automatically. Histogram metadata
records its contributing `time_bin_count`, `bintime` and
`acquisition_duration`. Supply `bintime` with an explicit matrix when
`dark_counts` is non-zero; without this provenance, `fit_lifetime()` rejects
the fit instead of applying a correction derived from an unrelated trace.

## Thresholding into on/off states

```python
states = ptu.threshold(off=0.1, on=0.6)   # {"on": Signal, "off": Signal}
```

`on`/`off` are fractions of the time-trace maximum. Time bins with counts
`>= on * max` contribute to the `"on"` histogram, bins `< off * max` to
`"off"`. Calling `threshold()` with defaults (`off=0`, `on=1`) puts the whole
matrix into `"on"`. Each histogram records the number and duration of the
selected bins, so lifetime fitting applies dark-count correction only for the
state that contributed to that histogram.

## Lifetime shift (salvaging a wrong sync offset)

If the sync offset of the acquisition was wrong, the decay starts late in the
lifetime window (e.g. the peak at 80 ns of a 100 ns period) and its tail
continues at the start of the window, after the next laser pulse. The fits
then only see the last 20 ns. Because the window spans exactly one laser
period, the data can be rotated back:

```python
ptu = PTU("scan.ptu", lifetime_shift="auto")   # peak moved to 5 ns
ptu.set_lifetime_shift(75.0)                   # or: shift to earlier times by 75 ns
ptu.set_lifetime_shift(None)                   # back to the raw data
```

`"auto"` puts the peak of the whole-file histogram 5 ns after zero
(`AUTO_SHIFT_MARGIN_NS`), leaving a baseline before the rise. Every photon's
lifetime bin becomes `(dtime - shift) mod period`, so all histograms, states,
fits, lifetime traces and FLIDs use the shifted data; cached traces are
rebuilt. The raw data is kept, and the applied shift is stored in
`ptu.lifetime_shift_ns` and in the metadata of the results.

Notes:

- This is a salvage tool; fix the sync offset of the acquisition instead.
- It requires the window to cover a full period (`rep_time / resolution`
  bins, at most 4096); otherwise it raises `ValueError`.
- If the period is not a whole number of bins (e.g. 1562.5), the partly
  covered last bin ends up inside the decay. It is left out of all lifetime
  fits (`ptu.lifetime_seam_bin`, `fit["seam_ns"]`).
- Time traces are unchanged. Photons moved across the window edge strictly
  belong to the previous pulse (one period, ~100 ns, earlier), which is
  negligible for time bins of milliseconds.

## Lifetime fitting

```python
fit = ptu.fit_lifetime(states["on"], n_exp=2,
                       mode="manual", start=0.1, end=70.0)
```

Fits a histogram with `n_exp` exponentials plus the dark-count offset. The fit
window's left bound is chosen by `mode`:

| `mode` | Left bound |
|--------|-----------|
| `"fraction"` (default) | rising edge reaching `fraction` (default `0.95`) of the smoothed max |
| `"inflection"` | dominant inflection point of the decay |
| `"manual"` | keep `start <= t <= end` (ns), `end=None` = last sample |

Additional keywords: `smoothing` (Gaussian sigma for edge finding),
`normalize` (scale the fitted curve peak to 1), `weighting` (see below),
`diagnostics` (thresholds of the over-fitting warnings, see below),
`fit_background` and `prerise`/`prerise_gap` (see
[Background and the bins before the rise](#background-and-the-bins-before-the-rise)),
plus any
[`MultiExpFit`](../src/kpfm_pl/fitting.py) keyword (`p0`, `bounds`, `loss`,
`f_scale`, `weights`, `constrain_sum_A`, ...). `NaN` entries of `p0` are
filled with the automatic guess, which is also clipped into `bounds`. With
`weighting="poisson"` a robust loss's `f_scale` is in Poisson standard
deviations.

| `weighting` | Fit statistic |
|---|---|
| `"none"` (default) | unweighted least squares |
| `"poisson"` | least squares with every bin weighted by its photon-counting noise `1/sqrt(max(counts, 1))` (Neyman χ²) |
| `"mle"` | Poisson maximum likelihood (minimises the Poisson deviance). Least squares underestimates lifetimes where bins hold few photons (long tails, per-bin fits); the MLE does not. Works with `normalize`; needs the linear `loss` |

On the test file `ptu_test1_…` the tail holds about 10 counts per bin; a
simulation with its decay gives a long lifetime of 28.6 ns with `"poisson"`
but 30.3 ns with `"mle"` (true: 30.2 ns).

```python
# tau_1 between 0.5 and 3 ns, everything else automatic, Poisson weights
fit = ptu.fit_lifetime(hist, n_exp=2, weighting="poisson",
                       bounds=([0, 0, 0.5, 1e-10], [np.inf, np.inf, 3.0, np.inf]))
```

The returned dict contains:

| Key | Meaning |
|-----|---------|
| `A`, `tau` | per-pulse amplitudes (corrected for the incomplete decay, see below) and lifetimes (ns) |
| `A_err`, `tau_err` | their standard errors (from `MultiExpFit.corrected_covariance()`) |
| `A_fit`, `A_fit_err` | the fitted amplitudes of the curve (uncorrected) and their errors |
| `tau_int`, `tau_int_err` | intensity-weighted mean lifetime `sum(A_i tau_i^2) / sum(A_i tau_i)` of the per-pulse amplitudes (ns, equal to `tau` for `n_exp=1`) and its propagated standard error; `tau_int_fit` is the uncorrected value |
| `tau_amp`, `tau_amp_err` | amplitude-weighted mean lifetime `sum(A_i tau_i) / sum(A_i)` and its error; `tau_amp_fit` uncorrected |
| `tau_mean`, `tau_mean_err`, `mean_lifetime` | the mean chosen with `mean_lifetime=` (`"intensity"` = `tau_int`, default, or `"amplitude"` = `tau_amp`) |
| `f_amp`, `f_amp_err` | amplitude fractions `A_i / sum(A)` (0..1) and their errors |
| `f_int`, `f_int_err` | intensity fractions `A_i tau_i / sum(A tau)` (0..1, share of the photons) and their errors |
| `period`, `decay_correction` | laser period `T` (ns) used for the incomplete-decay correction (`None` with `decay_correction=False`) |
| `offset`, `offset_err` | constant offset of the fitted data and its standard error (0 when fixed) |
| `background`, `background_err`, `fit_background` | the offset in counts per bin (also with `normalize`), its error, and whether it was fitted |
| `curve` | model evaluated on the fitted grid |
| `quality` | `{residuals, cost, success, message, r_squared}` plus the Poisson goodness of fit `{reduced_chi_squared, chi_squared, dof, deviance, weighted_residuals}` and the model selection `{bic, bic_statistic, n_points, n_params}` |
| `warnings` | list of over-fitting warnings (empty if none) |
| `weighting`, `statistic` | the `weighting` used and the resulting `MultiExpFit` statistic |
| `label` | LaTeX legend string of the fitted model |
| `success` | whether the optimiser converged |
| `shifted_histogram_x` | full lifetime axis shifted so the peak is at 0 ns |
| `t_fit`, `y_fit` | the time/intensity arrays actually fitted |
| `t_model` | time since the laser pulse of every fitted point (equal to `t_fit`, except `t_fit + T` for the bins before the rise) |
| `prerise`, `prerise_points`, `prerise_gap_ns` | whether the bins before the rise were fitted, how many (they are the first entries of `t_fit`), and the gap used |
| `left_bound_ns` | left bound of the fit window after the maximum (ns) |

**Fit quality.** `R²` is close to 1 for almost any decay, so it is a poor
measure for photon-counting data. `quality["reduced_chi_squared"]`
([`poisson_chi_squared`](../src/kpfm_pl/fitting.py), computed on the raw counts
with `sigma_i = sqrt(max(counts_i, 1))`) is about 1 when the model describes
the histogram within photon noise; clearly larger values mean systematic
deviations (too few exponentials, a badly placed window, ...).
`quality["weighted_residuals"]` (on `t_fit`) shows where they are; plotted
below the decay they should scatter around 0 without structure.

**Uncertainties.** The parameter covariance is `(J^T J)^-1` of the fit
Jacobian, scaled by the residual variance `sum(r^2) / dof` (the default of
`scipy.optimize.curve_fit`), so a poor fit also gets larger errors. The error
of `tau_int` is propagated from it, including the correlations between the
amplitudes and lifetimes. Amplitudes and lifetimes of a multi-exponential fit
are strongly correlated, so `tau_int` is usually much better defined than the
individual `tau_i`.

**Incomplete decay.** The laser fires every `T` (the sync period). If the
decay has not reached the noise floor within one period, every bin also
holds photons of all earlier pulses. For an exponential they add up to
`exp(-t/tau) / (1 - exp(-T/tau))`, so the fitted **lifetimes are correct**
but the fitted amplitudes are too large, the long components most (at
`T = 100 ns`: 0.7 % for `tau = 20 ns`, 16 % for `tau = 50 ns`). That would
push `tau_int` towards the long components. `fit_lifetime` (and the per-bin
fits of `lifetime_trace`/`flid`) therefore report per-pulse amplitudes
`A = A_fit * (1 - exp(-T/tau))` by default and compute the mean lifetimes and
fractions from them; the errors are propagated. The curve (`curve`, `label`)
uses the fitted `A_fit`. Pass `decay_correction=False` to report the fitted
amplitudes unchanged.

**Mean lifetime and fractions.** For several exponentials,
`tau_int = sum(A tau^2) / sum(A tau)` weights each component by its photons
and equals the mean photon arrival time (comparable with
[`arrival_time_trace`](#mean-arrival-time-no-fit)); `tau_amp = sum(A tau) /
sum(A)` weights by the amplitudes, i.e. the share of emitters (common for
quenching and FRET). `f_amp` and `f_int` give the same two views per
component. `mean_lifetime="amplitude"` makes `tau_mean`, the lifetime trace
and the FLID maps use `tau_amp`. The helpers
[`mean_lifetime`, `amplitude_fractions`, `intensity_fractions`](../src/kpfm_pl/fitting.py)
work on any amplitudes, also row-wise on `lifetime_trace(...).meta`.

**How many exponentials? (BIC and warnings)** Adding an exponential always
lowers χ², even if it only fits noise. `quality["bic"]` is the Bayesian
information criterion `BIC = χ² + k ln n` (Poisson deviance instead of χ²
for `weighting="mle"`), with `k = 2 n_exp` free parameters (one more with
`fit_background`) and `n` fitted bins. The lower BIC is preferred; compare only fits of the **same histogram,
window and weighting**. Kass & Raftery (1995) grade a difference of 2–6 as
positive, 6–10 as strong and > 10 as very strong evidence. `out["warnings"]`
([`fit_warnings`](../src/kpfm_pl/fitting.py)) flags typical signs of too many
components: an amplitude compatible with zero (below `amp_sigma`=2 standard
errors) or carrying less than `min_fraction`=1 % of the intensity, two
lifetimes that differ by less than `merge_sigma`=2 combined errors or
`merge_rel`=10 %, and a parameter within `bound_rel`=1 % of a bound. With a
fitted background it also flags a lifetime above `period_fraction`=50 % of
the laser period (see below). Change the thresholds with
`diagnostics={"min_fraction": 0.02, ...}`. References:
G. Schwarz, *Ann. Statist.* 6, 461 (1978); R. E. Kass & A. E. Raftery,
*J. Am. Stat. Assoc.* 90, 773 (1995); for MLE vs. least squares M. Maus et al.,
*Anal. Chem.* 73, 2078 (2001). Background on the fit statistics and the BIC:
[irf_reconvolution.md, sections 3–4](irf_reconvolution.md#3-fitting-statistics).

```python
fits = {n: ptu.fit_lifetime(hist, n_exp=n, weighting="mle") for n in (1, 2, 3)}
for n, fit in fits.items():
    print(n, fit["quality"]["bic"], fit["warnings"])
```

Plotting the fit (from `example_ptu.py`):

```python
fig, ax = plt.subplots()
ax.plot(fit["shifted_histogram_x"], states["on"].y, "bo", label="Data")
ax.plot(fit["t_fit"], fit["curve"], "r-", label=fit["label"])
ax.set_xlabel("Time (ns)"); ax.set_ylabel("Counts")
```

### Background and the bins before the rise

```python
fit = ptu.fit_lifetime(hist, n_exp=2, weighting="mle",
                       fit_background=True, prerise=True)   # both default to False
print(fit["background"], fit["background_err"], fit["prerise_points"])
```

Both options belong to the tail fit and are off by default.

**`fit_background=True`** fits the constant offset `C ≥ 0` instead of fixing
it at the dark counts. Use it when the histogram sits on a pedestal that the
dark counts do not cover: detector afterpulsing, stray light, or emission
much slower than the laser period. A fixed offset that is too low is not
harmless: the fit then puts the pedestal into its longest lifetime. The
result is `fit["background"] ± fit["background_err"]` in counts per bin; the
BIC counts it as one more parameter, so fits with and without it can be
compared on the same bins.

**`prerise=True`** also fits the bins before the rising edge. With periodic
excitation those bins hold the tail of the previous laser pulse, so a bin at
`t < 0` continues the decay at `t + T`. They enter the fit at that time
(`fit["t_model"]`; `fit["t_fit"]` keeps their position on the plot axis),
which extends the observed decay from the fit window to the whole laser
period. That is useful for samples whose decay has not reached the
background within one period. The model itself does not change: the tail fit
already fits the steady-state curve whose amplitudes include all earlier
pulses (see *Incomplete decay* above). `prerise_gap` (ns, default 2) is how
far before the histogram maximum the region ends, which keeps the rising
edge itself out of the fit; set it to a few widths of the instrument
response. The first `fit["prerise_points"]` entries of `t_fit`, `y_fit` and
`curve` are these bins, so break a plotted line there:

```python
n = fit["prerise_points"]
ax.plot(fit["t_fit"][:n], fit["curve"][:n], "r-")     # before the rise
ax.plot(fit["t_fit"][n:], fit["curve"][n:], "r-")     # the fit window
```

It needs no full-period lifetime window (`T` is the sync period) and is
ignored with `irf=`, where the whole period is fitted anyway.

Notes:

- **Bins outside the period.** `prerise` leaves out lifetime bins that are
  not fully inside one laser period: the bin past the period, a partly
  covered last bin, and bins after the last one that holds a photon anywhere
  in the file (legacy files whose header period is longer than the real
  one). `fit_background` alone keeps the bins of the plain tail fit, so the
  BIC of the two can be compared.
- **Long lifetimes and the background are hard to tell apart.** Within one
  period a slow decay is nearly flat: at `τ = T/2` it only falls to 14 %.
  The bins before the rise extend the time range by the part of the period
  before the maximum, which helps but does not remove the ambiguity. With a
  fitted background, `fit["warnings"]` therefore reports every lifetime above
  half the laser period (`diagnostics={"period_fraction": 0.5}`). In that
  case fix the background at an independently measured level (e.g. the count
  rate while the laser is off) or record with a lower repetition rate.
- On the test file `ptu_test1_…` (2 exponentials, `weighting="mle"`) the
  dark-count offset of 0.5 counts per bin gives 6.1 / 29.2 ns; with
  `fit_background=True` the fit finds 10.8 ± 0.3 counts per bin and
  2.5 / 14.1 ns, with a BIC of 1580 instead of 2010. Which is right depends
  on the sample: check what the offset should be before trusting either.

**Per time bin** (`lifetime_trace`, `flid`): pass `prerise=True` as well. A
time bin holds too few photons to fit its own background, so the per-bin
offset is never fitted; pass the background of the summed-histogram fit as
`background_rate=fit["background"] / hist.meta["acquisition_duration"]`
(counts per second and lifetime bin). Without it the dark counts are used.

### IRF reconvolution

The tail fit above ignores the instrument response function (IRF): laser
pulse width, timing jitter and above all the detector's transit-time spread.
It is fine when every lifetime is much longer than the IRF width, but short
components and the amplitudes are then biased. A reconvolution fit convolves
the model with a **measured IRF** and fits the whole laser period, including
the rising edge. How it works, which statistics to use, how to record a good
IRF and the literature are explained in
[irf_reconvolution.md](irf_reconvolution.md); this section covers the usage.

```python
irf = PTU("irf_scatter.ptu")          # scattered laser light, same detector and settings
fit = ptu.fit_lifetime(hist, n_exp=2, irf=irf, weighting="mle",
                       fit_background=True)          # optional: fit_scatter=True
print(fit["tau"], fit["shift_ns"], fit["background"], fit["quality"]["bic"])
```

The model on one period of `N` bins is

```
m_j = Σ_i a_i [g_δ ⊛ k_τi]_j + s·g_δ,j + B
```

| Symbol | Meaning |
|---|---|
| `g` | the IRF histogram minus its background (`irf_background`, default `"auto"` = mean of the floor bins below `median + 5 sqrt(median)`), normalised to unit area. The floor keeps its zero-mean noise: clipping it at 0 would leave a positive floor that the fit takes from `B` |
| `g_δ` | the IRF shifted later by `δ` (sub-bin, Fourier shift theorem); `δ` is fitted (`irf_shift=` fixes it) and corrects drift and the wavelength-dependent detector response ("colour effect") |
| `k_τ` | exponential kernel integrated over the bins; with `decay_correction=True` (default) it is periodic and includes the decay of all earlier pulses, otherwise a single pulse |
| `⊛` | circular (periodic) or linear convolution, computed by FFT |
| `a_i`, `τ_i` | amplitudes (counts per bin at the excitation time) and lifetimes of the IRF-free decay; `tau_int`, `tau_amp`, `f_amp`, `f_int` follow from them |
| `B` | constant background per bin: fixed from the dark counts, or fitted with `fit_background=True` |
| `s` | scattered laser light (counts, shaped like the IRF), with `fit_scatter=True` |

Requirements: the IRF file must have the **same TCSPC resolution and laser
period** (checked; a mismatch is an error) and the lifetime window must
cover a **whole laser period** (otherwise an error; choose a coarser
resolution). `mode="fraction"`/`"inflection"` fit the whole period except
the partial last bin; `mode="manual"` fits `start ≤ t ≤ end`. `normalize`
is not available. All `weighting`s work; `"mle"` is recommended.

The result adds `shift_ns`/`shift_err_ns`, `background`/`background_err`,
`scatter`/`scatter_err`, `irf_fwhm_ns`, `irf_background`, `irf_shifted`
(unit-area IRF after the shift) and `full_curve` on `irf_t_ns` (the period's
time axis), `reconvolution: True`; `quality["n_params"]` (and so the BIC)
counts `δ`, `B` and `s`. `PTU.irf_histogram(irf)` returns the validated IRF
histogram. The fit itself is
[`IRFReconvolutionFit`](../src/kpfm_pl/fitting.py), usable on any pair of
histograms (helpers `prepare_irf`, `exp_kernel`, `irf_fwhm`).

**Per time bin** (`lifetime_trace`, `flid`): pass `irf=` too. Per-bin fits
have few photons, so take the shared parameters from the fit of the summed
histogram: `irf_shift=fit["shift_ns"]`, `background_rate=fit["background"] /
hist.meta["acquisition_duration"]` (counts per second and lifetime bin) and
`scatter_fraction=fit["scatter"] / hist.y.sum()`; then only `a_i` and `τ_i`
are fitted per bin.

Record the IRF with the same detector, filters, resolution, sync settings
and cable delays as the sample, at a low count rate and with many counts;
scattered laser light or, for SPADs with a strong colour effect, a quenched
dye emitting near the sample's wavelength. Literature: J. R. Lakowicz,
*Principles of Fluorescence Spectroscopy*, 3rd ed., ch. 4 (2006); W. Becker,
*The bh TCSPC Handbook*; S. C. Warren et al., *PLoS ONE* 8, e70687 (2013)
(FLIMfit's model); M. Maus et al., *Anal. Chem.* 73, 2078 (2001) (MLE). The
full explanation with all references is in
[irf_reconvolution.md](irf_reconvolution.md).

## Lifetime trace

```python
tt = ptu.time_trace(bintime=0.5, laser_pulse=False)
lt = ptu.lifetime_trace(bintime=0.5, n_exp=2, bins=10)
```

`lifetime_trace` fits every photon time bin and returns a `Signal` containing
the fitted lifetime in ns: for `n_exp > 1`, the intensity-weighted mean
`sum(A_i tau_i^2) / sum(A_i tau_i)` (or the amplitude-weighted
`sum(A_i tau_i) / sum(A_i)` with `mean_lifetime="amplitude"`), as in the
[FLID map](#flid-map). `decay_correction` works as in `fit_lifetime`;
`prerise`/`prerise_gap` and `background_rate` are described in
[Background and the bins before the rise](#background-and-the-bins-before-the-rise).
Its `x`
and `y` arrays have exactly one entry per selected time bin. Bins outside an
intensity ROI, empty bins, failed fits and bins whose amplitudes are all zero
remain `NaN` at their original positions; the individual fitted components
are available in `Signal.meta["taus"]` and `Signal.meta["amplitudes"]`
(per-pulse amplitudes, corrected for the incomplete decay as in
[`fit_lifetime`](#lifetime-fitting); the uncorrected ones are in
`Signal.meta["amplitudes_fit"]`).

When `laser_pulse` is omitted, `lifetime_trace` inherits the mode used by the
most recent `time_trace` call. This keeps legacy `laser_pulse=False` traces on
the same centred, zero-padded bins. Pass `laser_pulse=True` or `False`
explicitly when creating a lifetime trace without first creating its matching
PL time trace.

Like `flid`, `lifetime_trace` takes an optional `progress(done, total)`
callback that reports progress and can cancel the fit (see
[Progress and cancelling](#flid-map)), and checks the photons per time bin
(`min_counts`, `coverage`, `auto_bintime`, see
[Photons per time bin](#photons-per-time-bin)).

### Mean arrival time (no fit)

```python
arrival = ptu.arrival_time_trace(bintime=0.5)    # same arguments as lifetime_trace
```

A fast sanity check of the fitted lifetimes. For each time bin it takes the
mean delay of the photons inside the fit window (chosen exactly as for the
fits, on the summed histogram of the selected bins). For a decay
`sum(A_i exp(-t/tau_i))` that mean is exactly the intensity-weighted mean
lifetime `sum(A_i tau_i^2) / sum(A_i tau_i)` of the fits, wherever the window
starts. Two effects bias it, and both are corrected:

- **Truncation.** The window has a finite length `T` (e.g. 100 ns at 10 MHz,
  minus the part before the peak), so late photons are missing and the mean
  is too short: `tau - T / (exp(T/tau) - 1)` for one exponential. That is
  negligible for `tau < T/7`, about 3 % at `tau = T/5` and 22 % at
  `tau = T/2.5`. `truncation_correction=True` (default) inverts this relation
  ([`utils.lifetime_from_truncated_mean`](../src/kpfm_pl/utils.py)); it is
  exact for one exponential and approximate for several.
- **Dark counts.** A flat background has the mean delay `T/2`, independent
  of `tau`, so a background fraction `f` gives `(1 - f) tau + f T/2`. With
  ~100 photons per bin even a few background photons matter, so the expected
  `dark_counts * bintime / n_lt_bins` photons per lifetime channel are
  subtracted first. Set `dark_counts` correctly.

`y` is the (corrected) lifetime in ns; `meta` holds `mean_arrival` (the
uncorrected mean), `photons` (dark-corrected photons in the window),
`window_ns`, `window_start_ns`, `dark_per_channel` and, as for
`lifetime_trace`, `counts`, `roi`, `bintime` and `count_check`.

A clear difference between the two traces is informative: on a
multi-exponential decay a single-exponential fit follows the short component,
while the mean arrival time follows the intensity-weighted mean, so the fit
needs more exponentials.

### Photons per time bin

`lifetime_trace`, `arrival_time_trace` and `flid` check that enough of the
selected time bins hold enough photons for a reliable lifetime:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `min_counts` | `100` | photons a time bin should hold; `None` or `0` disables the check |
| `coverage` | `0.9` | fraction of the selected bins that must reach `min_counts` (`0.5` = the median) |
| `auto_bintime` | `False` | recompute with the suggested bin time when the check fails |

A failed check raises a `RuntimeWarning` with the suggested bin time:
`bintime * min_counts / level`, rounded up to 1, 2, 2.5 or 5 × 10ⁿ s, where
`level` is the photon number reached by `coverage` of the bins (counts grow
linearly with the bin time). With `auto_bintime=True` the call is repeated
with that bin time, up to three more times, since blinking makes counts grow
more slowly than the bin time. The ROI's count range is scaled by the same
factor, because it was drawn at the original bin time. It stops with a warning
when fewer than two bins would be left. The result is stored in
`meta["count_check"]` (`passed`, `level`, `fraction`, `suggested_bintime`,
`requested_bintime`, ...). `meta["bintime"]` and `meta["roi"]` hold the values
actually used.

## FLID map

```python
flid = ptu.flid(roi={"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 500},
                bintime=0.05, n_exp=1)      # -> Image
```

A FLID (Fluorescence-Lifetime-Intensity-Distribution) fits each time bin of
the (optionally ROI-restricted) matrix and bins the resulting
`(lifetime, counts)` pairs into a 2-D histogram (lifetime on `x` [ns], counts
on `y`, occurrence as `data`).

`flid` is `lifetime_trace` followed by `flid_map` (see
[Redrawing a map](#redrawing-a-map-without-fitting-again)).
`method="mean_arrival"` uses `arrival_time_trace` instead, without fitting
(`n_exp`, `bins`, `weighting` and fit keywords are then ignored, `isolated` is
not available; `truncation_correction` is passed on).

For `n_exp > 1` the lifetime of a time bin is the **intensity-weighted mean**
of its fitted components,

$$\tau_\text{mean} = \frac{\sum_i A_i \tau_i^2}{\sum_i A_i \tau_i},$$

which reduces to `tau` for `n_exp=1`. Bins whose amplitudes are all zero are
left out. `isolated=True` skips the averaging and returns one map per
component instead.

Key parameters:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `roi` | `None` | `{"x_min","x_max","y_min","y_max"}`; overrides the explicit `x_*`/`y_*` bounds |
| `x_min`/`x_max`/`y_min`/`y_max` | `None` | manual bounds (time in s, counts on y) |
| `bintime` | `0.05` | time-bin width (s) |
| `n_exp` | `1` | exponentials per per-bin fit |
| `bins` | `1` | lifetime-axis reduction factor used before each fit |
| `lifetime_bins` | `100` | histogram bins on the fitted-lifetime axis (1–1024) |
| `intensity_bins` | `100` | histogram bins on the raw-count axis (1–1024) |
| `lifetime_range`/`intensity_range` | `None` | `(min, max)` of the histogram axes (ns / counts); values outside are dropped. `None` = data range padded by `ceil(3·sigma)` bins per side (never below 0) |
| `mode`/`fraction`/`smoothing`/`start`/`end` | – | fit-window selection, as in `fit_lifetime` |
| `sigma` | `1.5` | Gaussian smoothing of the 2-D histogram (`0` disables) |
| `isolated` | `False` | when `True`, return one `Image` per exponential; otherwise a single map of the intensity-weighted mean lifetime |
| `weighting` | `"none"` | `"poisson"` weights every per-bin fit by photon noise, `"mle"` fits each bin by Poisson maximum likelihood (recommended for few photons per bin), as in `fit_lifetime` (also for `lifetime_trace`) |
| `mean_lifetime` | `"intensity"` | mean lifetime of a bin with `n_exp > 1`: `"intensity"` (`tau_int`) or `"amplitude"` (`tau_amp`); fitted maps only (also for `lifetime_trace`) |
| `decay_correction` | `True` | correct the per-bin amplitudes for the incomplete decay of earlier pulses, as in `fit_lifetime` (also for `lifetime_trace`) |
| `prerise`/`prerise_gap` | `False`/`2.0` | also fit the bins more than `prerise_gap` ns before the histogram maximum in the per-bin tail fits, as in `fit_lifetime` (also for `lifetime_trace`; fitted maps only) |
| `background_rate` | `None` | constant offset of the per-bin fits in counts per second and lifetime bin, e.g. from `fit_lifetime(..., fit_background=True)`; `None` = dark counts (also for `lifetime_trace`; fitted maps only) |
| `method` | `"fit"` | `"mean_arrival"`: map the mean photon arrival time instead of fitting |
| `truncation_correction` | `True` | for `method="mean_arrival"`, see [Mean arrival time](#mean-arrival-time-no-fit) |
| `min_counts`/`coverage`/`auto_bintime` | `100`/`0.9`/`False` | photon-count check, see [Photons per time bin](#photons-per-time-bin) |
| `progress` | `None` | callback `progress(done, total)`, see below |

**Progress and cancelling.** Fitting every time bin can take a while (seconds
to minutes, depending on `bintime`). `flid` and `lifetime_trace` accept an
optional `progress(done, total)` callback. It is called before each per-bin fit
and once with `done == total` at the end. An exception raised inside the
callback propagates out of the call, which cancels the computation:

```python
def report(done, total):
    if done % 500 == 0:
        print(f"{done}/{total} bins fitted")

flid = ptu.flid(bintime=0.05, progress=report)
```

Per-bin `taus`, `amplitudes` (fitted `A_i`, both shape `(time bins, n_exp)`)
and `counts` are stored in `Image.meta`. `counts` contains the raw
photon total in each time bin; lifetime curves are normalised only for fitting.
The two explicit histogram sizes keep FLID memory bounded independently of the
measurement duration.

**Histogram ranges.** By default each map spans its own data, padded so the
Gaussian smoothing (`sigma`) is not cut off at the outermost points. Two FLIDs
computed this way have different bins, so their occurrence values are not
directly comparable. To compare FLIDs (e.g. several ROIs of one trace), pass
the same `lifetime_range` and `intensity_range` to each call. The ranges
actually used are stored in `meta["lifetime_range"]` and
`meta["intensity_range"]`. Together they are the exact image edges:

```python
extent = (*flid.meta["lifetime_range"], *flid.meta["intensity_range"])
plt.imshow(flid.data, extent=extent, origin="lower", aspect="auto")
```

`flid.extent` spans only the first to last bin *centre*, which is half a bin
short on every side.

### Redrawing a map without fitting again

The slow part of a FLID is the per-bin lifetime; the map is a cheap 2-D
histogram of it. `flid_map` builds the map from a `lifetime_trace` or
`arrival_time_trace` `Signal`, so other histogram settings, or a fitted and a
mean-arrival map on identical bins, need no new fits:

```python
from kpfm_pl import flid_map

lt = ptu.lifetime_trace(roi=roi, bintime=0.5)
arrival = ptu.arrival_time_trace(roi=roi, bintime=0.5)
ranges = dict(lifetime_range=(0, 20), intensity_range=(0, 4000))
fitted = flid_map(lt, sigma=1.5, **ranges)
check = flid_map(arrival, sigma=1.5, **ranges)     # same bins: directly comparable
smoother = flid_map(lt, sigma=3.0, **ranges)       # no new fits
```

`flid_map` takes `lifetime_bins`, `intensity_bins`, `lifetime_range`,
`intensity_range`, `sigma`, `isolated` and an optional `label`. The map's
`meta` copies the trace's `meta` and adds `time`, `bin_lifetimes` (the mapped
value of every time bin), the histogram settings and the ranges actually used.

`get_roi_matrix(roi, bintime)` returns the raw photon count matrix filtered by
an ROI's time window and count threshold (useful to build custom histograms).

## Laser triggers and markers

```python
triggers = ptu.laser_triggers(marker_channel=4, reference=tt)
markers  = ptu.get_markers(marker_channel=4)   # all marker times (s)
```

`laser_triggers` returns **every** marker on the channel as a chronological
list of trigger points `[{"index", "time"}, ...]` – the format consumed by
[`Match`](match.md). A single marker gives a one-point list. Without a hardware
marker it falls back to the two time-trace edges stored by
`time_trace(laser_pulse=False)` (legacy files). It warns and returns `None` if
no trigger is found.

Indices and times are mapped onto `reference` (default: the most recent time
trace), so `reference.x[point["index"]] == point["time"]`. Markers outside the
reference axis are dropped with a warning rather than clamped to an unrelated
endpoint.

`get_markers` is deliberately simple: it returns the exact hardware marker
times (s) on the channel as an array, useful for debugging. These exact times
can also be passed straight to `Match`.
