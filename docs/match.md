# Match: correlating a PL time trace with a KPFM signal (`kpfm_pl.Match`)

`Match` takes two [`Signal`](../src/kpfm_pl/core.py) traces – a KPFM lock-in
signal and a photoluminescence (PL) time trace – plus the laser triggers of
each, and:

1. **shifts** the time axes so the *left* laser triggers coincide,
2. **amplitude-scales** the two signals with a selectable method,
3. **resamples** both onto a shared, uniform time grid.

It then offers statistical helpers (Pearson correlation, cross-correlation /
lag) over the trigger-bounded overlap region. There is no plotting inside the
class – everything is returned as `Signal`/`dict` objects.

The runnable example is
[`examples/example_correlate.py`](../examples/example_correlate.py), which uses
a current PTU recording with marker triggers. The separate
[`examples/example_correlate_legacy.py`](../examples/example_correlate_legacy.py)
demonstrates the legacy `laser_pulse=False` fallback.

## End-to-end example

```python
import kpfm_pl as k

# PL time trace (PTU). Current recordings use the default laser_pulse=True,
# with triggers read from the marker channel. See docs/ptu.md for the legacy
# laser_pulse=False fallback.
ptu = k.PTU("scan.ptu", dark_counts=2.25)
tt = ptu.time_trace(bintime=0.05)
tt_triggers = ptu.laser_triggers()

# KPFM signal (Zurich); map triggers onto the averaged axis via reference=
z = k.Zurich("scan.txt")
streams = z.stream_names()
kpfm = z.average(streams[0], bintime=0.05)
z_triggers = z.laser_triggers(streams[1], reference=kpfm, method="slope")

m = k.Match(kpfm, tt,
            kpfm_triggers=z_triggers,
            timetrace_triggers=tt_triggers,
            kpfm_smoothing=20, timetrace_smoothing=1)

k_sig = m.kpfm_signal()        # aligned KPFM Signal on the shared grid
t_sig = m.timetrace_signal()   # aligned time trace on the shared grid
print("Pearson:", m.pearson())
print("Best lag (s):", m.cross_correlation()["best_lag"])
```

## Constructor parameters

```python
Match(kpfm, timetrace, *, kpfm_triggers, timetrace_triggers,
      n_points=None, kpfm_smoothing=1, timetrace_smoothing=1,
      scaling="none", trigger_window=5)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `kpfm`, `timetrace` | – | the two `Signal` traces to align |
| `kpfm_triggers`, `timetrace_triggers` | – | trigger dicts (`{"left": {"index","time"}, "right": ...}`) from the loaders, or a single `float` time |
| `n_points` | `None` | samples in the shared grid; `None` uses the finer native resolution over the overlap; an explicit value must be an integer of at least 2 |
| `kpfm_smoothing`, `timetrace_smoothing` | `1` | moving-average window applied to each signal before alignment |
| `scaling` | `"none"` | amplitude-scaling method (see below) |
| `trigger_window` | `5` | half-width (samples) of the median filter on each trigger anchor when `scaling="trigger"` |

A single `float` trigger sets the right trigger equal to the left (only the
time-shift/resampling is well defined) and disallows `scaling="trigger"`.
With a single trigger there is no laser-on window, so the
`"feature"`/`"zscore"` statistics (and the trigger-bounded statistics below)
fall back to the full overlap.

## Scaling methods

| `scaling` | Effect |
|-----------|--------|
| `"none"` (default) | only shift + resample; y axes, and units are preserved |
| `"feature"` | min-max scale each signal to `[0, 1]`; output y units are dimensionless (`"1"`) |
| `"zscore"` | standardize each signal to zero mean / unit std; output y units are dimensionless (`"1"`) |
| `"trigger"` | quantitative two-point calibration mapping the PL trace onto the KPFM target scale; both output y units match the KPFM input unit |

For `"feature"`/`"zscore"` the statistics are taken over the laser-on
(trigger-bounded) window so the laser-off baseline cannot dominate. For
`"trigger"` the KPFM signal is left untouched and the time trace is anchored to
the KPFM values at the two triggers (its natural rising orientation is
preserved via `abs()` of the scale).

## Outputs

```python
m.kpfm_signal(on_grid=True)       # aligned KPFM Signal
m.timetrace_signal(on_grid=True)  # aligned, scaled time trace Signal
m.to_dict()                       # both signals on the grid + metadata
m.kpfm_derivative(smoothing=1)    # dΦ/dt of the aligned KPFM signal
```

- `kpfm_signal` / `timetrace_signal` return the aligned signals on the shared
  grid (`on_grid=True`) or their native axes (`on_grid=False`). The trigger
  locations on the common (KPFM) reference frame are stored in the
  loader-compatible form `{"left": {"index", "time"}, "right":
  {"index", "time"}}`. Indices refer to the common grid, and
  `grid[point["index"]] == point["time"]`.
- Each output records the original label, units, and metadata in
  `signal["meta"]["source"]`.
- `to_dict()` bundles `grid`, both signals, the scaling method, the per-signal
  `scale`/`shift` and `kpfm_scale`/`kpfm_shift`, and the trigger times.

## Statistics

```python
m.pearson(within_triggers=True)
# -> {"statistic": r, "pvalue": p}

m.cross_correlation(within_triggers=True, normalize=True)
# -> {"lag": <s array>, "correlation": <array>, "best_lag": <s>,
#     "best_correlation": <float>}
```

Both restrict to the laser-on window between the triggers by default. For the
cross-correlation, a **positive** `best_lag` means the time trace leads the
KPFM signal (the KPFM signal lags); a **negative** `best_lag` means the time
trace lags. Statistics use only paired finite samples and raise `ValueError`
when fewer than two pairs remain or either signal is constant. Construction
also rejects empty, non-finite, or non-increasing time axes and signals with no
overlapping time span.

## Plotting tip

The aligned signals share the same time grid, so a twin-axis plot works well.
From the example, the trigger lines come straight from the signal metadata:

```python
k_sig = m.kpfm_signal()
ax.axvline(k_sig["meta"]["triggers"]["left"]["time"], color="k", linestyle="--")
ax.axvline(k_sig["meta"]["triggers"]["right"]["time"], color="k", linestyle="--")
```
