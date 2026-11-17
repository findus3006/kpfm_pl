# PTU files: time traces, lifetimes and photon correlations (`kpfm_pl.PTU`)

`PTU` reads PicoQuant `.ptu` files (TCSPC / photoluminescence, T3 or T2
mode) and gives you the photon **time trace**, **lifetime histograms**,
**lifetime fits**, the **lifetime over time**, **FLID maps** and the
**second-order correlation** of two detectors. Results are `Signal` (curves)
or `Image` (maps) objects, ready for matplotlib; fits return a dictionary.

This manual is organised by task. The method behind the lifetime fits (fit
window, statistics, background, mean lifetimes, number of components) is
explained in [lifetime_fitting.md](lifetime_fitting.md), the fit with a
measured instrument response in [irf_reconvolution.md](irf_reconvolution.md).
A runnable example is [`examples/example_ptu.py`](../examples/example_ptu.py)
with `ptu_test2_LaserPulse(...).ptu` from the `Test data/` folder.

```python
from kpfm_pl import PTU

ptu = PTU("scan.ptu", dark_counts=2.25)
trace = ptu.time_trace(bintime=0.05)              # photons per 50 ms
hist = ptu.lifetime_histogram()                   # summed over the trace
fit = ptu.fit_lifetime(hist, n_exp=2)             # Poisson maximum likelihood
print(fit["tau"], fit["tau_int"], fit["quality"]["reduced_chi_squared"])
```

## Opening a file

```python
ptu = PTU("scan.ptu", dark_counts=2.25)
```

The file is read completely on opening, which takes a few seconds for large
files; a line reports the number of photons and markers.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `filepath` | – | the `.ptu` file |
| `dark_counts` | `0.0` | detector dark counts per second. They are the fixed background of the lifetime fits and are subtracted before mean arrival times are taken, so set them correctly |
| `lifetime_shift` | `None` | rotate the lifetime axis of a recording with a wrong sync offset, see [Lifetime shift](#lifetime-shift-salvaging-a-wrong-sync-offset) |
| `channels` | `None` | detector(s) to analyse, see [Detectors](#detectors); `None` = all detectors together |
| `lifetime_binning` | `None` | time units combined into one lifetime bin, see [Record formats](#record-formats-t3-and-t2) |
| `sync_divider` | `None` | T2 files with laser pulses: laser pulses per recorded sync event (`None` = from the sync rate in the file header) |
| `smooth_rep_rate` | `False` | T3 files: use the laser period from the header rounded to its leading digit (e.g. 100 ns) |

What the file tells you after opening:

| Attribute | Meaning |
|-----------|---------|
| `acquisition_time` | measuring time (s) |
| `rep_time` | laser period (s); measured from the sync events for T2 files |
| `resolution` | width of one lifetime bin (s) |
| `n_lt_bins`, `lt_time` | number of lifetime bins and their times (s) |
| `measurement_mode`, `device` | `"T3"` or `"T2"` and the instrument family |
| `has_lifetime` | whether the photons carry a delay after a laser pulse (`False` for T2 files without laser pulses) |
| `channels`, `selected_channels` | detectors that recorded photons, and the ones in use (`None` = all) |
| `timetag_resolution` | time-tag unit of a T2 file (s) |
| `is_cw` | continuous-wave T3 file (see below) |
| `marker` | hardware markers per marker channel (use [`get_markers`](#laser-triggers-and-markers) for times in s) |
| `tags` | all header entries |

**Continuous-wave files** (a T3 file without laser sync) cannot be sorted
into lifetime bins. The file opens with a notice, `is_cw` is `True` and every
time-binned or lifetime analysis raises a clear error.

## Record formats (T3 and T2)

The file header says how the photons were recorded; nothing has to be chosen
by hand.

| Device | T3 | T2 |
|--------|----|----|
| PicoHarp 300 | yes | yes |
| HydraHarp (V1 and V2), TimeHarp 260 N/P | yes | yes |
| MultiHarp, PicoHarp 330 | yes | yes |

- **T3**: every photon is stored with the number of its laser pulse and its
  delay after that pulse. This is what lifetime analysis needs.
- **T2**: every event is stored with an absolute time tag.
  - **With laser pulses** (the sync input, channel 0 of the MultiHarp /
    HydraHarp family, carries the pulse train): the delay of each photon is
    its time since the last laser pulse, so the file behaves like a T3 file.
    Everything in this manual works; the laser period is measured from the
    sync events.
  - **Without laser pulses** (PicoHarp 300: both inputs are detectors; or a
    sync input that is not connected): there is no lifetime information
    (`has_lifetime` is `False`). Time traces, markers, laser triggers,
    [`Match`](match.md) and the [second-order correlation](#second-order-correlation-g2-antibunching)
    work; lifetime methods raise an error that says so.

```python
ptu = PTU("multiharp_t2.ptu")     # prints the format, period and lifetime bins
ptu.measurement_mode              # "T2"
ptu.has_lifetime                  # True: laser pulses were recorded
```

**Lifetime bins.** T2 delays (and T3 delays of the MultiHarp family) come in
units of a few ps, far finer than needed. By default the smallest power of
two of these units is combined that leaves at most 4096 lifetime bins per
laser period; `ptu.lifetime_binning` and `ptu.resolution` give the result.
PicoHarp 300 T3 files keep their native bins. Pass `lifetime_binning=` to
choose yourself.

**Sync divider.** If only every N-th laser pulse is recorded, the delays are
folded back to one period. N is taken from the sync rate in the file header;
pass `sync_divider=N` if it is missing. Periods without sync events (laser
sync lost for a while) are handled the same way, and photons before the first
recorded pulse are kept.

Notes:

- A T2 file records every laser pulse as an event (40 MB per second at
  10 MHz without a divider), so such files get large.
- Sync events that are not periodic (the sync input used for something else)
  are not taken as laser pulses; a warning says so and the file has no
  lifetimes.
- T2 reading has been tested on synthetic files only. Check `ptu.rep_time`
  and the lifetime histogram of your first real T2 file, in particular when
  the MultiHarp used a sync divider.

## Detectors

Detectors are numbered from 1 (`ptu.channels`). By default the photons of all
detectors are used together, as from a single detector. To analyse one:

```python
ptu = PTU("scan.ptu", channels=2)     # only detector 2
ptu.set_channels(None)                # all detectors again
ptu.set_channels([1, 2])              # several, summed
```

The selection applies to every time trace, lifetime histogram, fit, lifetime
trace and FLID; a lifetime shift is applied again to the selection. Markers,
laser triggers and the second-order correlation do not depend on it. Results
record the selection (`meta["channels"]`, `fit["channels"]`).

The arrival time of every photon is available at full resolution,
independent of the selection:

```python
t1 = ptu.photon_times(1)              # s, detector 1
t2 = ptu.photon_times(2)
ptu.photon_channel                    # detector of every photon (record order)
```

T2 files give the time tags; T3 files the laser pulse number times the period
plus the delay after the pulse.

## Looking at the time trace

```python
tt = ptu.time_trace(bintime=0.05)     # Signal: x = time (s), y = photons per bin
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `bintime` | `0.05` | width of one time bin (s) |
| `tstart`, `tend` | `0.0`, `None` | time window; `None` = to the end of the file |
| `laser_pulse` | `True` | how the laser triggers were recorded, see below |

The window is `[tstart, tend)`; every bin that overlaps it is kept, so the
first `x` can lie slightly before `tstart`. `x` holds the left edge of each
bin.

`laser_pulse` refers to the **pulse generator in the laser driver** (e.g. a
10 MHz laser sends a pulse every 100 ns, which the PicoHarp uses as its sync):

- `True` (default, and how data is recorded now): the pulse generator runs
  throughout the file, whether or not the laser light is on. The laser on/off
  triggers are recorded on a marker channel (see
  [Laser triggers](#laser-triggers-and-markers)).
- `False`: a work-around for old files without markers, where the pulse
  generator only ran while the laser was on. The non-zero part of the trace
  is centred and zero-padded, and its two edges are stored as triggers in
  `tt.meta["triggers"]`. Only use it for such files.

`tt.meta` holds the photon matrix (time bins × lifetime bins, whose row sums
are `tt.y`), the bin durations and the average count rate.

The trace is remembered by the reader: `lifetime_histogram()`, `threshold()`
and `laser_triggers()` without arguments work on the most recent one, and
lifetime traces and FLIDs replace it. Call `time_trace` again before those
three if you have computed something else in between.

## Choosing photons: on/off states, time windows, ROIs

**On and off states** of a blinking emitter:

```python
states = ptu.threshold(off=0.1, on=0.6)   # {"on": Signal, "off": Signal}
fit_on = ptu.fit_lifetime(states["on"], n_exp=2)
```

`on` and `off` are fractions of the largest count of the current trace. Bins
with at least `on × max` counts form the on-state histogram, bins below
`off × max` the off state. With the defaults (`off=0`, `on=1`) everything
is in "on". Each histogram records how many bins and how much time it
contains, so the dark-count background is right for each state.

**A time window** is simply the window of the trace: `time_trace(bintime,
tstart, tend)` followed by `lifetime_histogram()`.

**Regions of interest (ROIs)** combine a time window with a count range:

```python
roi = {"x_min": 10.0, "x_max": 60.0, "y_min": 200, "y_max": 800}  # s, counts per bin
matrix = ptu.get_roi_matrix(roi, bintime=0.05)                   # selected time bins
hist = ptu.lifetime_histogram(matrix, bintime=0.05)
```

The count range belongs to the bin time it was chosen at. The same `roi`
dictionary is accepted by the lifetime trace, the FLID and the second-order
correlation.

**The lifetime histogram** of any selection:

```python
hist = ptu.lifetime_histogram()                       # the current trace
hist = ptu.lifetime_histogram(matrix, bintime=0.05)   # explicit photon matrix
```

It is a `Signal` of counts against the lifetime-bin time (s). Pass
`bintime` with an explicit matrix when `dark_counts` is set: the dark-count
background needs the measuring time of the selection, and the fit refuses a
histogram without it.

## Measuring a lifetime

```python
fit = ptu.fit_lifetime(hist, n_exp=2)
print(fit["tau"], fit["tau_err"])           # lifetimes (ns)
print(fit["tau_int"], fit["tau_int_err"])   # intensity-weighted mean lifetime (ns)
```

Without an IRF this is a **tail fit**: `n_exp` exponentials plus a constant
background are fitted to the decay after its maximum, where the instrument
response has died away. `t = 0` is at the histogram maximum. Why, and what
it means for the amplitudes, is in
[lifetime_fitting.md](lifetime_fitting.md#2-the-tail-fit).

### The fit window

| `mode` | The window starts |
|--------|-------------------|
| `"fraction"` (default) | after the maximum, where the smoothed decay has fallen to `fraction` (default `0.95`) of it |
| `"inflection"` | at the steepest point of the falling edge (the maximum if there is none) |
| `"manual"` | at `start` and ends at `end` (ns from the maximum; `end=None` = last bin) |

`smoothing` (default 2 bins) is the Gaussian smoothing used to find the
edge.

### The fit statistic

| `weighting` | Statistic |
|---|---|
| `"mle"` (default) | Poisson maximum likelihood: correct also where bins hold few photons (long tails) |
| `"poisson"` | least squares weighted by the photon noise `1/√counts` |
| `"none"` | plain least squares |

See [lifetime_fitting.md, section 5](lifetime_fitting.md#5-counting-statistics-how-the-bins-are-weighted)
for the difference. Three situations replace the maximum likelihood, each
with a warning; `fit["weighting"]` always states the statistic actually
used:

- a robust `loss` (e.g. `loss="soft_l1"` against afterpulsing spikes)
  exists only for least squares, so the fit uses `"poisson"`;
- explicit `weights` make it plain least squares with those weights;
- if the maximum likelihood does not converge (typically a badly specified
  model, e.g. a fixed background far below the real one), the
  Poisson least-squares result is reported (`fit["mle_converged"]` is
  `False`, an `MLEFallbackWarning` is issued). Its BIC then comes from χ² and
  is not compared with maximum-likelihood fits.

### Background and the bins before the rise

```python
fit = ptu.fit_lifetime(hist, n_exp=2, fit_background=True, prerise=True)
print(fit["background"], fit["background_err"], fit["prerise_points"])
```

By default the background is fixed at the dark counts. `fit_background=True`
fits it instead: for a pedestal the dark counts do not explain (stray light,
afterpulsing, emission slower than the laser period). The result is
`fit["background"] ± fit["background_err"]` in counts per bin. With a fitted
background every lifetime above half the laser period is reported in
`fit["warnings"]`, because such a slow decay and a constant are barely
separable. See [lifetime_fitting.md, section 4](lifetime_fitting.md#4-the-background).

`prerise=True` also fits the bins more than `prerise_gap` ns (default 2)
before the maximum. They hold the tail of the previous laser pulse and enter
the fit at their time since that pulse, `t + T`, which extends the observed
decay to almost the whole period: useful for decays that have not reached
the background within one period. They are the first
`fit["prerise_points"]` entries of `t_fit`, `y_fit` and `curve`; break a
plotted line there:

```python
n = fit["prerise_points"]
ax.plot(fit["t_fit"][:n], fit["curve"][:n], "r-")     # before the rise
ax.plot(fit["t_fit"][n:], fit["curve"][n:], "r-")     # the fit window
```

`prerise` does not apply to the IRF reconvolution, which fits the whole
period anyway.

### Mean lifetimes, fractions and the incomplete decay

The fit reports per-pulse amplitudes `A`: the fitted ones (`A_fit`) contain
the photons of earlier laser pulses that had not decayed yet and are too
large for long lifetimes, `A = A_fit · (1 − exp(−T/τ))`. All mean lifetimes
and fractions use `A`; the lifetimes need no correction.
`decay_correction=False` reports the fitted amplitudes unchanged.

| Result | Meaning |
|---|---|
| `tau_int` | intensity-weighted mean `Σ Aτ² / Σ Aτ` (mean photon arrival time) |
| `tau_amp` | amplitude-weighted mean `Σ Aτ / Σ A` (e.g. for quenching and FRET) |
| `f_int` | intensity fractions `Aᵢτᵢ / Σ Aτ` (share of the photons) |
| `f_amp` | amplitude fractions `Aᵢ / Σ A` (share of the emitters) |
| `tau_mean` | the mean chosen with `mean_lifetime="intensity"` (default) or `"amplitude"` |

All come with standard errors (`_err`), propagated from the fit including the
correlations between the parameters. See
[lifetime_fitting.md, sections 3 and 8](lifetime_fitting.md#3-periodic-excitation).

### Is the fit good, and how many components?

- `fit["quality"]["reduced_chi_squared"]`, computed with Poisson weights on
  the raw counts, is about 1 when the model describes the data within photon
  noise.
- `fit["quality"]["weighted_residuals"]` (on `t_fit`) should scatter around 0
  without structure.
- `fit["quality"]["bic"]` compares fits with different `n_exp` on the **same
  histogram, window and statistic**; the lower value wins (a difference above
  10 is very strong evidence).
- `fit["warnings"]` lists signs of too many components: an amplitude
  compatible with zero or below 1 % of the photons, two merged lifetimes, a
  parameter at a bound. The thresholds can be changed with
  `diagnostics={"min_fraction": 0.02, ...}` (keys `amp_sigma`,
  `min_fraction`, `merge_sigma`, `merge_rel`, `bound_rel`,
  `period_fraction`).

```python
fits = {n: ptu.fit_lifetime(hist, n_exp=n) for n in (1, 2, 3)}
for n, fit in fits.items():
    print(n, fit["quality"]["bic"], fit["warnings"])
```

Background: [lifetime_fitting.md, sections 6 and 7](lifetime_fitting.md#6-is-the-fit-good).

### Starting values and bounds

Initial guesses and bounds are passed as `p0` and `bounds`, laid out as
`[A1, …, An, τ1, …, τn]` (plus the background with `fit_background`).
`NaN` entries of `p0` are filled automatically.

```python
# tau_1 between 0.5 and 3 ns, everything else automatic
fit = ptu.fit_lifetime(hist, n_exp=2,
                       bounds=([0, 0, 0.5, 1e-10], [np.inf, np.inf, 3.0, np.inf]))
```

Other options: `loss`/`f_scale` (robust loss; `f_scale` in units of the
photon noise with Poisson weighting), `normalize` (fit the histogram scaled
to its peak; the quality is still computed on the raw counts).

### Fitting with a measured IRF (reconvolution)

Record the instrument response (IRF), e.g. scattered laser light, with the
same detector, filters, TCSPC resolution, sync settings and cable delays as
the sample. Then:

```python
irf = PTU("irf_scatter.ptu")
fit = ptu.fit_lifetime(hist, n_exp=2, irf=irf, fit_background=True)  # optional: fit_scatter=True
print(fit["tau"], fit["shift_ns"], fit["background"], fit["quality"]["bic"])
```

The model decay is convolved with the IRF and fitted over the whole laser
period, rising edge included. Fitted are the amplitudes and lifetimes of the
IRF-free decay, the IRF shift δ (`irf_shift=` fixes it, in ns), optionally
the background (`fit_background=True`; otherwise the dark counts) and
scattered laser light shaped like the IRF (`fit_scatter=True`). The
amplitudes refer to the excitation time, so the fractions and mean lifetimes
do not depend on a choice of `t = 0`.

Requirements: the IRF file must have the same TCSPC resolution and laser
period, and the lifetime window must cover a whole laser period (otherwise
an error; choose a coarser resolution). The automatic modes fit the whole
period, `mode="manual"` fits `start ≤ t ≤ end`. `normalize` is not
available. The method, how to record a good IRF and the literature are in
[irf_reconvolution.md](irf_reconvolution.md).

Extra results: `shift_ns`/`shift_err_ns`, `background`/`background_err`,
`scatter`/`scatter_err`, `irf_fwhm_ns`, `irf_shifted` (the unit-area IRF
after the shift) and `full_curve` on `irf_t_ns` (the whole period).
`ptu.irf_histogram(irf)` returns the IRF histogram after the checks.

### Seam bins

Some lifetime bins are not fully inside one laser period and do not follow
the decay:

- bins past the period, which never fill;
- the last bin of the period when the period is not a whole number of bins:
  it is only partly covered (period 1603.70 bins: the last one collects about
  70 % of the counts of its neighbours);
- bins after the last one that ever holds a photon (old files whose header
  period is longer than the real one).

They are found from this geometry, not from the counts, and left out of every
lifetime fit, lifetime trace, mean arrival time and FLID by default
(`exclude_seam=True`; with lifetime binning the whole group holding one).
They sit at the end of the window, or inside the decay after a
[lifetime shift](#lifetime-shift-salvaging-a-wrong-sync-offset).
`ptu.seam_bins()` returns the mask; a fit lists them in `fit["excluded_ns"]`
(positions on `shifted_histogram_x`), `fit["seam_points"]` and
`fit["seam_ns"]` (the partly covered bin).

### Plotting a fit

```python
fig, ax = plt.subplots()
ax.semilogy(fit["shifted_histogram_x"], hist.y, "o", ms=2, label="Data")
ax.semilogy(fit["t_fit"], fit["curve"], "r-", label=fit["label"])
ax.set_xlabel("Time after the maximum (ns)"); ax.set_ylabel("Counts")
```

### Fit results

| Key | Meaning |
|-----|---------|
| `A`, `A_err`, `tau`, `tau_err` | per-pulse amplitudes (counts per bin) and lifetimes (ns) with standard errors |
| `A_fit`, `A_fit_err` | amplitudes of the drawn curve (not corrected for the incomplete decay) |
| `tau_int`, `tau_amp`, `tau_mean` (± `_err`) | mean lifetimes, see above; `tau_int_fit`, `tau_amp_fit` from the uncorrected amplitudes |
| `f_int`, `f_amp` (± `_err`) | intensity and amplitude fractions (0…1) |
| `period`, `decay_correction` | laser period used for the correction (ns) and whether it was applied |
| `background`, `background_err`, `fit_background` | background in counts per bin, its error, and whether it was fitted |
| `weighting`, `statistic`, `mle_converged` | statistic actually used, and whether the maximum likelihood converged (`None` for least squares) |
| `quality` | `reduced_chi_squared`, `chi_squared`, `dof`, `deviance`, `weighted_residuals`, `bic`, `n_points`, `n_params`, `r_squared`, … |
| `warnings` | signs of over-fitting (empty if none) |
| `curve`, `label` | model on the fitted points and a legend text |
| `shifted_histogram_x` | the whole lifetime axis in ns, 0 at the maximum |
| `t_fit`, `y_fit`, `t_model` | fitted times, counts, and the model times (`t + T` for the bins before the rise) |
| `left_bound_ns` | start of the fit window (ns) |
| `prerise`, `prerise_points`, `prerise_gap_ns` | bins before the rise: used, how many, gap |
| `exclude_seam`, `excluded_ns`, `seam_points`, `seam_ns` | seam bins, see above |
| `lifetime_shift_ns`, `channels` | applied lifetime shift and detector selection |
| `success` | whether the fit converged |

## Lifetime shift (salvaging a wrong sync offset)

If the sync offset of a recording was wrong, the decay starts late in the
lifetime window (e.g. the peak at 80 ns of a 100 ns period) and its tail
continues at the start of the window, after the next laser pulse. The fits
would only see the last 20 ns. Because the window spans exactly one laser
period, the delays can be rotated back:

```python
ptu = PTU("scan.ptu", lifetime_shift="auto")   # peak moved to 5 ns
ptu.set_lifetime_shift(75.0)                   # or: shift to earlier times by 75 ns
ptu.set_lifetime_shift(None)                   # back to the raw data
```

`"auto"` puts the peak of the whole-file histogram 5 ns after zero, which
leaves a baseline before the rise. Every photon's delay becomes
`(delay − shift) mod period`, so all histograms, states, fits, lifetime
traces and FLIDs see the shifted data. The raw data is kept; the applied
shift is `ptu.lifetime_shift_ns` and is recorded in the results.

Notes:

- It is a salvage: fix the sync offset of the setup instead.
- It needs a window that covers a whole laser period (at most 4096 bins);
  otherwise it raises an error.
- Time traces do not change. Strictly, photons moved across the window edge
  belong to the previous pulse, about 100 ns earlier, which does not matter
  for time bins of milliseconds.
- **Lifetime shift or bins before the rise?** Both use that a bin before the
  rise holds the tail of the previous pulse. The shift rotates the **data**,
  so every histogram, plot, fit and FLID sees one continuous decay; it is
  meant for a peak late in the window. The
  [bins before the rise](#background-and-the-bins-before-the-rise) leave the
  data alone and only give those bins the time `t + T` in the fit that asks
  for them. They can be combined.

**The shift is in whole bins, the period is not.** The rotation moves the
delays by whole lifetime bins, but the laser period is usually not a whole
number of bins. On `ptu_test2` the period is 1603.70 bins and the histogram
has 1604. A bin that does not cross the window edge moves by exactly the
shift, so its time is correct. A bin that wraps around is placed 1604 bins
later, but physically it belongs 1603.70 bins later: it is the previous
pulse's decay, one period on. After a shift, every bin behind the seam
therefore sits 0.30 bin (19 ps) too late relative to the bins before it.
The error is constant and lies between zero and one bin depending on the
file (31 ps on `test3`).

A time offset δ on part of the decay looks like a step in amplitude of δ/τ
at the seam: 0.1 % for τ = 20 ns, 1 % for τ = 2 ns. On the sample files the
shifted fit can be compared with the unshifted fit including the bins before
the rise, which uses the exact times of the same data:

| File | Shifted (whole bins) | Exact times |
|---|---|---|
| `ptu_test2` | 20.183 / 4.039 ns | 20.181 / 4.039 ns |
| `test3` | 26.511 / 4.024 ns | 26.509 / 4.023 ns |

That is about 0.01 %, far below the fit uncertainty. It would only matter
for lifetimes well below 1 ns at the seam or for very high photon numbers.
No correction is applied. A correction would leave the data and the plot as
they are and give the wrapped bins their exact time in the fit model, as the
bins before the rise do. It would also depend on the header period being
right, so it could not help old files whose header period is too long (on
`ptu_test1` by five bins).

## Following the lifetime over time: the lifetime trace

```python
tt = ptu.time_trace(bintime=0.5)
lt = ptu.lifetime_trace(bintime=0.5, n_exp=1)    # Signal: lifetime (ns) per time bin
```

Every time bin is fitted on its own, with the same fit window as
`fit_lifetime` (chosen on the summed histogram of the selected bins). `y` is
the lifetime in ns, for several components the mean lifetime chosen with
`mean_lifetime` (intensity-weighted by default). There is one value per time
bin; bins outside an ROI, empty bins and failed fits are `NaN`. The single
components are in `lt.meta["taus"]` and `lt.meta["amplitudes"]` (per-pulse;
`amplitudes_fit` uncorrected).

**Statistic.** The per-bin fits use Poisson least squares by default
(`weighting="poisson"`): from about 100 photons per bin it gives nearly the
same lifetimes as the maximum likelihood and is several times faster.
`weighting="mle"` is available; bins where it does not converge get the
Poisson value, are marked in `lt.meta["mle_fallback"]` and are counted in
one warning.

**Shared parameters.** A time bin holds too few photons for its own
background, so the background is fixed: at the dark counts, or at
`background_rate` (counts per second and lifetime bin), e.g. from a fit of
the summed histogram with a fitted background:

```python
summed = ptu.fit_lifetime(hist, n_exp=2, fit_background=True)
rate = summed["background"] / hist.meta["acquisition_duration"]
lt = ptu.lifetime_trace(bintime=0.5, n_exp=2, background_rate=rate)
```

With `irf=` every bin is fitted by reconvolution; take the IRF shift,
background and scatter from the fit of the summed histogram:
`irf_shift=fit["shift_ns"]`, `background_rate=…` as above and
`scatter_fraction=fit["scatter"] / hist.y.sum()`. Then only amplitudes and
lifetimes are fitted per bin.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `roi` or `x_min`/`x_max`/`y_min`/`y_max` | `None` | time bins to fit, see [ROIs](#choosing-photons-onoff-states-time-windows-rois) |
| `bintime` | `0.05` | width of the time bins (s) |
| `n_exp`, `bins` | `1`, `1` | components per bin, and lifetime bins combined before fitting |
| `mode`, `fraction`, `smoothing`, `start`, `end` | – | fit window, as in `fit_lifetime` |
| `weighting` | `"poisson"` | statistic of the per-bin fits |
| `mean_lifetime`, `decay_correction` | `"intensity"`, `True` | as in `fit_lifetime` |
| `prerise`, `prerise_gap` | `False`, `2.0` | bins before the rise, as in `fit_lifetime` |
| `background_rate` | `None` | fixed background (counts per second and lifetime bin); `None` = dark counts |
| `irf`, `irf_shift`, `scatter_fraction` | `None` | per-bin reconvolution |
| `exclude_seam` | `True` | leave the [seam bins](#seam-bins) out |
| `laser_pulse` | as the last time trace | see [time trace](#looking-at-the-time-trace) |
| `min_counts`, `coverage`, `auto_bintime` | `100`, `0.9`, `False` | photon-count check, see below |
| `progress` | `None` | callback `progress(done, total)`; raise from it to cancel |

### Photons per time bin

A lifetime from a time bin with `N` photons scatters by about `τ/√N` from
photon noise alone. The lifetime trace, the mean arrival time and the FLID
therefore check that enough bins hold enough photons:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `min_counts` | `100` | photons a bin should hold; `None` or `0` switches the check off |
| `coverage` | `0.9` | fraction of the selected bins that must reach `min_counts` (`0.5` = the median) |
| `auto_bintime` | `False` | recompute with the suggested bin time when the check fails |

A failed check warns and suggests a bin time (`bintime × min_counts / level`,
rounded up to 1, 2, 2.5 or 5 × 10ⁿ s, where `level` is the photon number
reached by `coverage` of the bins). With `auto_bintime=True` the calculation
is repeated with that bin time, up to three more times, since blinking makes
the counts grow more slowly than the bin time. The count range of an ROI is
scaled along. The outcome is in `meta["count_check"]`; `meta["bintime"]` and
`meta["roi"]` hold the values actually used.

### Mean arrival time (no fit)

```python
arrival = ptu.arrival_time_trace(bintime=0.5)    # same selection arguments as lifetime_trace
```

A fast, fit-free cross-check. For each time bin it takes the mean delay of
the photons in the fit window. For one exponential that is exactly `τ`,
wherever the window starts. For several components it is the
intensity-weighted mean of the decay **at the window start**, longer than
the fitted `τ_int` because the short components have already faded there.
Two effects are corrected:

- **Finite window.** The window has a length `T` (about the laser period
  minus the part before the peak), so late photons are missing:
  `τ − T/(exp(T/τ) − 1)` for one exponential, negligible below `τ = T/7`,
  3 % at `T/5`, 22 % at `T/2.5`. `truncation_correction=True` (default)
  inverts this; exact for one exponential, approximate for several.
- **Dark counts.** A flat background has a mean delay of `T/2`, so a few
  background photons in a bin of 100 matter. The expected dark counts per
  lifetime bin are subtracted first; set `dark_counts` correctly.

`y` is the corrected lifetime in ns; `meta` holds `mean_arrival` (uncorrected),
`photons` (dark-corrected photons in the window), `window_ns`,
`window_start_ns` and `dark_per_channel`. A clear difference to the fitted
trace shows that one exponential per bin is not enough.

## Lifetime–intensity maps (FLID)

```python
roi = {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 500}
flid = ptu.flid(roi, bintime=0.05, n_exp=1)      # Image: lifetime (ns) × counts
```

A FLID (fluorescence-lifetime–intensity distribution) fits every time bin,
as the lifetime trace does, and collects the pairs (lifetime, counts) in a
two-dimensional histogram: lifetime on `x` (ns), counts per bin on `y`,
occurrence as `data`. It shows, for example, whether dim states of a
blinking emitter are also short-lived. `method="mean_arrival"` maps the mean
arrival time instead of a fit (no `n_exp`, `bins` or `weighting`).

For several components the lifetime of a bin is the mean chosen with
`mean_lifetime`; `isolated=True` returns one map per component instead.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `roi` or `x_min`/`x_max`/`y_min`/`y_max` | `None` | time bins to use; all `None` = the whole trace |
| `bintime`, `n_exp`, `bins`, fit-window, `weighting`, `mean_lifetime`, `decay_correction`, `prerise`, `background_rate`, `irf`…, `exclude_seam`, `min_counts`… | – | per-bin fits, as for the [lifetime trace](#following-the-lifetime-over-time-the-lifetime-trace) |
| `lifetime_bins`, `intensity_bins` | `100`, `100` | histogram bins on the lifetime and the count axis (1–1024) |
| `lifetime_range`, `intensity_range` | `None` | `(min, max)` of the axes (ns, counts); values outside are dropped. `None`: the data range, padded so the smoothing is not cut off |
| `sigma` | `1.5` | Gaussian smoothing of the map in map bins (`0` = none) |
| `isolated` | `False` | one map per component |
| `method`, `truncation_correction` | `"fit"`, `True` | `"mean_arrival"`: no fit, see [Mean arrival time](#mean-arrival-time-no-fit) |
| `progress` | `None` | callback `progress(done, total)`; raise from it to cancel |

```python
def report(done, total):
    if done % 500 == 0:
        print(f"{done}/{total} bins fitted")

flid = ptu.flid(roi, bintime=0.05, progress=report)
```

**Smoothing and the width of the map.** Even an emitter with one fixed
lifetime gives a spread of fitted lifetimes from photon noise, about `τ/√N`:
the typical funnel, wide at low and narrow at high counts. The smoothing adds
its own width in quadrature: with 0.1 ns per map bin, `sigma=1.5` adds
0.15 ns, which at `τ = 3 ns` widens the cloud by about 12 % at 100 photons
but doubles it at 1000 photons. A smoothed map can therefore suggest a
heterogeneity that is not there. Judge the width with `sigma=0` (redraw, see
below).

**Comparing maps.** By default each map spans its own data, so two maps have
different bins and their occurrences cannot be compared directly. Pass the
same `lifetime_range` and `intensity_range` to both. The ranges used are in
`meta["lifetime_range"]` and `meta["intensity_range"]` and are the exact
image edges:

```python
extent = (*flid.meta["lifetime_range"], *flid.meta["intensity_range"])
plt.imshow(flid.data, extent=extent, origin="lower", aspect="auto")
```

(`flid.extent` spans only the first to the last bin centre, half a bin short
on every side.) `meta` also holds the per-bin `taus`, `amplitudes` and
`counts` (raw photons per time bin).

### Redrawing a map without fitting again

The slow part is fitting the time bins; the map is a cheap histogram of the
results. `flid_map` builds the map from a lifetime trace or mean-arrival
trace, so other histogram settings, or a fitted and a mean-arrival map on
identical bins, need no new fits:

```python
from kpfm_pl import flid_map

lt = ptu.lifetime_trace(roi, bintime=0.5)
arrival = ptu.arrival_time_trace(roi, bintime=0.5)
ranges = dict(lifetime_range=(0, 20), intensity_range=(0, 4000))
fitted = flid_map(lt, sigma=1.5, **ranges)
check = flid_map(arrival, sigma=1.5, **ranges)     # same bins: directly comparable
sharp = flid_map(lt, sigma=0, **ranges)            # no smoothing, no new fits
```

`flid_map` takes `lifetime_bins`, `intensity_bins`, `lifetime_range`,
`intensity_range`, `sigma`, `isolated` and `label`.

## Second-order correlation g2 (antibunching)

With a 50/50 beam splitter in front of two detectors (Hanbury Brown–Twiss
setup) the correlation of their photon streams shows whether the light
comes from a single quantum emitter. A single emitter cannot emit two photons
at once, so coincidences at zero delay are missing: **antibunching**.

```python
ptu = PTU("emitter_t2.ptu")        # T2 file, detectors on channels 1 and 2
g2 = ptu.g2()                      # Signal: coincidences against the lag
print(g2.meta["g2_0"], g2.meta["g2_0_err"])
```

Only **T2 files** are supported: they store the time tag of every photon at
the full timing resolution of the hardware. A T3 file raises an error.

**What is counted.** For every photon of detector `a` and every photon of
detector `b` within `±max_lag` of it, the time difference `τ = t_b − t_a` is
added to a histogram (all pairs, not only the next photon, so the
correlogram does not drop off at longer lags). Positive lags mean that the
photon of `b` came later. `g2.x` is the lag in s (bin centres, lag 0 in the
middle of a bin), `g2.y` the number of coincidences per lag bin.

**Normalisation.** `g2.meta["g2"]` is the correlogram divided by the
coincidences expected for uncorrelated light, `N_a N_b Δτ / T` (`N`: photons
per detector, `Δτ`: lag bin, `T`: duration of the selected photons). It is 1
for uncorrelated light at all lags.

**g2(0) with pulsed excitation.** The coincidences form peaks spaced by the
laser period: the side peaks hold pairs of photons from different laser
pulses, the centre peak pairs from the same pulse. `g2(0)` is the area of the
centre peak divided by the mean area of the side peaks:

| `g2(0)` | Meaning |
|---|---|
| 0 | ideal single-photon emitter |
| < 0.5 | a single emitter (two or more equal emitters give at least 0.5) |
| `1 − 1/N` | `N` equal, independent emitters |
| 1 | uncorrelated (classical) light, e.g. laser light |

The laser period comes from the laser sync of the file (MultiHarp, channel
0). A file without a sync (PicoHarp 300 T2) needs `period=` in s. Every side
peak whose whole window lies inside the lag range is used (with the default
`max_lag` of 5.5 periods: five on each side). The error is the Poisson error
of the areas.

**g2(0) with continuous excitation.** Without a period the correlogram has no
peaks; `g2(0)` is the normalised `g2` at the delay (the lag bin there, or the
mean within `±window`). Its dip has the width of the excited-state lifetime
(at low excitation), so choose a lag bin well below it.

**Which photons.** As for the FLID, the photons are chosen by time bins of a
trace, here the trace of the two detectors (`detector_trace`, width
`bintime`): the bins inside the time window `[x_min, x_max)` whose counts lie
in `[y_min, y_max)`. A count range picks the on state of a blinking emitter,
where the background of the off periods would otherwise fill the centre
peak; `roi=` takes a box drawn on the trace.

```python
trace = ptu.detector_trace(bintime=0.01, channels=(1, 2))    # photons per 10 ms
on = ptu.g2(bintime=0.01, y_min=0.6 * trace.y.max())        # the on state only
roi = ptu.g2({"x_min": 10, "x_max": 60, "y_min": 50, "y_max": 400}, bintime=0.01)
ptu.g2(period=100e-9)                                         # PicoHarp 300 T2: 10 MHz
```

`detector_trace` and `g2` neither use nor change the detector selection or
the time trace of the lifetime analysis.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `roi`, `x_min`, `x_max`, `y_min`, `y_max` | `None` | photons of the selected time bins (see above); all `None`: the whole file |
| `bintime` | `0.05` | width of the trace bins the region is chosen on (s); keep it much longer than the lags |
| `channels` | `(1, 2)` | the two detectors `(a, b)` |
| `max_lag` | 5.5 periods, else 500 ns | largest lag (s) |
| `lag_bin` | period / 200, else 1 ns | width of one lag bin (s), rounded to whole time tags |
| `period` | from the sync | laser period (s) for the pulsed `g2(0)`; `None` without a sync: continuous |
| `delay` | automatic | lag of the centre peak (s): detector and cable delays. Pulsed: found from the peak comb (all peaks folded onto one period), so it works even without a centre peak; continuous: 0 |
| `window` | half a period / half a lag bin | half-width of the area around each peak (pulsed) or around the delay (continuous) |
| `progress` | `None` | callback `progress(done, total)`; raise from it to cancel |

The result's `meta` holds `g2`, `g2_0`, `g2_0_err`, `method` (`"pulsed"` or
`"continuous"`), `centre_area`, `side_areas` (`{k: area}` for the side
peaks `k = ±1, ±2, …`), `mean_side_area`, `n_side_peaks` (pulsed), `period`,
`delay`, `window`, `lag_bin`, `max_lag`, `photons` and `rates` per detector,
`duration`, `selected_bins`, `roi` and `bintime`.

Notes:

- **Background.** Uncorrelated background photons (dark counts, stray light)
  add coincidences to every peak alike and raise `g2(0)` above the value of
  the emitter. The default window of half a period integrates the background
  between the peaks as well; a narrower `window` (a few lifetimes) leaves
  most of it out.
- **Edges of the region.** Photon pairs that cross from a selected into an
  unselected time bin are lost, about `max_lag / bintime` of the
  coincidences. That is negligible for bins of milliseconds and lags of
  microseconds.
- **Detector artefacts.** Dead time removes coincidences of one detector with
  itself only, which are not counted. Crosstalk between the two detectors
  (a breakdown flash of one detector seen by the other) shows as narrow
  peaks at a fixed lag next to zero.
- T2 reading is so far tested on synthetic files only (see
  [Record formats](#record-formats-t3-and-t2)).

The pair counting and `g2(0)` are also available on their own, for arrays
of integer time tags: `photon_correlation(a, b, max_lag, bin_width)`,
`peak_delay(lags, counts, period)` and
`pulsed_g2_zero(lags, counts, period, delay=, window=)`.

References: R. Hanbury Brown & R. Q. Twiss, *Nature* 177, 27 (1956);
B. Lounis & W. E. Moerner, *Nature* 407, 491 (2000); P. Michler et al.,
*Science* 290, 2282 (2000).

## Laser triggers and markers

The laser on/off triggers connect the PL trace with the KPFM signal
([`Match`](match.md)).

```python
triggers = ptu.laser_triggers(marker_channel=4, reference=tt)   # [{"index", "time"}, ...]
markers = ptu.get_markers(marker_channel=4)                       # all marker times (s)
```

`laser_triggers` returns every marker on the channel as a chronological list
of points `{"index", "time"}`, the format `Match` expects; one marker gives a
one-point list. Indices and times are placed on `reference` (default: the
most recent time trace), so `reference.x[point["index"]] == point["time"]`.
Markers outside it are dropped with a warning. Old files without markers fall
back to the two edges of a `time_trace(laser_pulse=False)`. Without any
trigger it warns and returns `None`.

`get_markers` returns the exact marker times in s, e.g. for checking; they
can also be passed to `Match` directly. Markers work the same for T3 and T2
files of every supported device (a PicoHarp 300 T2 marker is off by at most
60 ps).
