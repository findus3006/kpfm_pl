# Match: aligning a PL time trace with a KPFM signal (`kpfm_pl.Match`)

`Match` takes two [`Signal`](../src/kpfm_pl/core.py) traces – a KPFM lock-in
signal and a photoluminescence (PL) time trace – plus the laser triggers of
each, and:

1. **shifts** the time trace so one chosen trigger coincides with its KPFM
   counterpart,
2. **resamples** both signals onto a shared, uniform time grid that spans
   every data point of both signals,
3. **maps every KPFM trigger** onto that grid.

You then pick the region you care about with `extract`, either between two
triggers or between two times. There is no amplitude scaling, no statistics
and no plotting inside the class – everything is returned as `Signal`/`dict`
objects, ready for your own post-processing.

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

window = m.extract(m.triggers[0], m.triggers[-1])   # first to last trigger
k_sig, t_sig = window["kpfm"], window["timetrace"]
```

## Triggers

A trigger is a point `{"index": int, "time": float}`; a collection of triggers
is a chronological list of such points. This is what
[`PTU.laser_triggers`](ptu.md#laser-triggers-and-markers) and
[`Zurich.laser_triggers`](zurich.md#laser-triggers) return. `Match` also
accepts a single point, or plain time(s) in seconds – e.g. the exact hardware
marker times from `ptu.get_markers()`.

## Constructor parameters

```python
Match(kpfm, timetrace, *, kpfm_triggers, timetrace_triggers,
      align=0, dt=None, kpfm_smoothing=1, timetrace_smoothing=1)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `kpfm`, `timetrace` | – | the two `Signal` traces to align |
| `kpfm_triggers`, `timetrace_triggers` | – | a list of trigger points, a single point, or time(s) in seconds |
| `align` | `0` | position of the trigger in **both** lists used for alignment; negative values count from the end |
| `dt` | `None` | grid spacing in seconds; `None` uses the finer of the two native resolutions |
| `kpfm_smoothing`, `timetrace_smoothing` | `1` | moving-average window applied to each signal before alignment |

The time trace is shifted by
`offset = timetrace_triggers[align]["time"] - kpfm_triggers[align]["time"]`.
Both loaders report rising flanks by default, so the lists normally pair up
one-to-one. If they don't (e.g. a legacy file read with `flank="both"`), pick
the pair with `align`, or slice the lists before passing them.

## Grid and triggers

```python
m.offset     # time shift subtracted from the time-trace axis (s)
m.grid       # shared uniform grid (s), KPFM time frame
m.dt         # grid spacing (s)
m.triggers   # every KPFM trigger mapped onto m.grid
```

- The grid spans **all** samples of both signals. Where one signal has no data
  of its own, its grid values are `NaN`.
- The grid passes exactly through the aligned trigger. With `dt` equal to the
  PTU `bintime`, the shifted PTU bins land exactly on grid points, so the
  photon counts are carried over unchanged rather than interpolated.
- After alignment both signals live in the same frame, so there is one
  trigger list: `m.triggers`, taken from the KPFM triggers. Every point
  satisfies `m.grid[point["index"]] == point["time"]`.

## Outputs

```python
m.extract(start=None, stop=None)  # {"kpfm": Signal, "timetrace": Signal}
m.kpfm_signal(on_grid=True)       # full aligned KPFM Signal
m.timetrace_signal(on_grid=True)  # full aligned time trace Signal
m.to_dict()                       # both full signals on the grid + metadata
```

`extract` cuts both signals to an inclusive window on the shared grid. Each
bound is a time in seconds or a trigger point; `None` defaults to the start /
end of the region where **both** signals have data:

```python
m.extract(m.triggers[0], m.triggers[-1])  # same window as the legacy code
m.extract(10.0, 300.0)                    # from 10 s to 300 s
m.extract()                               # the full overlap
```

- `kpfm_signal` / `timetrace_signal` return the aligned signals on the shared
  grid (`on_grid=True`) or their native axes (`on_grid=False`, the time-trace
  axis shifted by `offset`).
- Every output keeps the source `label` and y `units`, and stores
  `meta["triggers"]` (the triggers inside that signal, indexed onto its own
  axis), `meta["time_shift"]` (`0` for KPFM, `offset` for the time trace),
  `meta["smoothing"]`, and the original label/units/metadata in
  `meta["source"]`.
- `to_dict()` bundles `grid`, both signals, `offset` and `triggers`.

Construction rejects empty, non-finite, or non-increasing time axes, invalid
triggers, and signals with no overlapping time span after the shift.

## Plotting tip

The extracted signals share the same time axis, so a twin-axis plot works
well. The trigger lines come straight from the signal metadata:

```python
w = m.extract(m.triggers[0], m.triggers[-1])
ax.plot(w["kpfm"]["x"], w["kpfm"]["y"])
for point in w["kpfm"]["meta"]["triggers"]:
    ax.axvline(point["time"], color="k", linestyle="--")
```
