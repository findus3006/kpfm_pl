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
`normalize` (scale the fitted curve peak to 1), plus any
[`MultiExpFit`](../src/kpfm_pl/fitting.py) keyword (`loss`, `f_scale`,
`weights`, `constrain_sum_A`, ...).

The returned dict contains:

| Key | Meaning |
|-----|---------|
| `A`, `tau` | fitted amplitudes and lifetimes (ns) |
| `offset` | constant offset used |
| `curve` | model evaluated on the fitted grid |
| `quality` | `{residuals, cost, success, message, r_squared}` |
| `label` | LaTeX legend string of the fitted model |
| `success` | whether the optimiser converged |
| `shifted_histogram_x` | full lifetime axis shifted so the peak is at 0 ns |
| `t_fit`, `y_fit` | the time/intensity arrays actually fitted |
| `left_bound_ns` | left bound of the fit window (ns) |

Plotting the fit (from `example_ptu.py`):

```python
fig, ax = plt.subplots()
ax.plot(fit["shifted_histogram_x"], states["on"].y, "bo", label="Data")
ax.plot(fit["t_fit"], fit["curve"], "r-", label=fit["label"])
ax.set_xlabel("Time (ns)"); ax.set_ylabel("Counts")
```

## Lifetime trace

```python
tt = ptu.time_trace(bintime=0.5, laser_pulse=False)
lt = ptu.lifetime_trace(bintime=0.5, n_exp=2, bins=10)
```

`lifetime_trace` fits every photon time bin and returns a `Signal` containing
the mean fitted lifetime in ns. Its `x` and `y` arrays have exactly one entry
per selected time bin. Bins outside an intensity ROI, empty bins and failed
fits remain `NaN` at their original positions; the individual fitted
components are available in `Signal.meta["taus"]`.

When `laser_pulse` is omitted, `lifetime_trace` inherits the mode used by the
most recent `time_trace` call. This keeps legacy `laser_pulse=False` traces on
the same centred, zero-padded bins. Pass `laser_pulse=True` or `False`
explicitly when creating a lifetime trace without first creating its matching
PL time trace.

## FLID map

```python
flid = ptu.flid(roi={"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 500},
                bintime=0.05, n_exp=1)      # -> Image
```

A FLID (Fluorescence-Lifetime-Intensity-Distribution) fits each time bin of
the (optionally ROI-restricted) matrix and bins the resulting
`(lifetime, counts)` pairs into a 2-D histogram (lifetime on `x` [ns], counts
on `y`, occurrence as `data`).

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
| `mode`/`fraction`/`smoothing`/`start`/`end` | – | fit-window selection, as in `fit_lifetime` |
| `sigma` | `1.5` | Gaussian smoothing of the 2-D histogram (`0` disables) |
| `isolated` | `False` | when `True`, return one `Image` per exponential; otherwise a single mean-lifetime map |

Per-bin `taus`/`counts` are stored in `Image.meta`. `counts` contains the raw
photon total in each time bin; lifetime curves are normalised only for fitting.
The two explicit histogram sizes keep FLID memory bounded independently of the
measurement duration.

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
