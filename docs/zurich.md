# Zurich reader (`kpfm_pl.Zurich`)

Parses Zurich Instruments lock-in `.txt` exports (KPFM surface potential,
laser-trigger channel, ...) into per-stream
[`Signal`](../src/kpfm_pl/core.py) objects. No plotting inside the loader –
return the arrays and plot them yourself.

The Zurich reader is used in
[`examples/example_correlate.py`](../examples/example_correlate.py)
together with a `.ptu` file, via the `Match` class.

## File format

The exporter writes a multi-stream text file: a `%`-prefixed header naming
each stream, followed by `time;value` data rows. All streams share the same
number of samples. Two quirks are handled automatically:

- The exported time axis is **negative and reversed**; it is flipped to
  forward time on load.
- Malformed rows are skipped with a warning; the total row count must be
  divisible by the number of streams or a `ValueError` is raised.

## Loading a file

```python
from kpfm_pl import Zurich

z = Zurich("scan.txt")
print(z.stream_names())     # e.g. ['Demod 6 Sample Auxiliary Input 1', ...]
```

After load, `z.streams` is a `{name: {"time", "value"}}` mapping and `z.time`
is the shared time axis of the first stream.

The lab-default stream names are exposed as module constants:

| Constant | Value | Role |
|----------|-------|------|
| `KPFM_STREAM` | `"Demod 6 Sample Auxiliary Input 1"` | KPFM surface potential |
| `TRIGGER_STREAM` | `"Demod 6 Sample Auxiliary Input 2"` | laser trigger channel |

## Getting a stream as a Signal

```python
raw = z.signal(stream=z.stream_names()[0])          # full-resolution Signal
kpfm = z.average(z.stream_names()[0], bintime=0.05)  # time-averaged Signal
```

- `signal(stream=KPFM_STREAM, units="V")` – return a stream at native
  resolution.
- `average(stream=KPFM_STREAM, bintime=0.05, units="V")` – bin-average into
  `bintime`-wide bins. The bin size is derived from the data's own sampling
  interval; the trailing partial bin is averaged over its valid samples.
  `meta` records `stream`, `bintime` and `averaged=True`.

## Laser triggers

```python
triggers = z.laser_triggers(z.stream_names()[1],
                            reference=kpfm, method="slope")
```

Locates the laser triggers in a trigger stream and returns them as a
chronological list of trigger points `[{"index": int, "time": float}, ...]` –
the format consumed by [`Match`](match.md). Returns `None` (with a warning) if
no trigger is found.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `stream` | `TRIGGER_STREAM` | trigger stream name |
| `method` | `"threshold"` | `"threshold"` min-max scales the signal and finds contiguous windows above `0.1`; the first sample of a window is its rising flank, the last sample its falling flank. `"slope"` finds flanks via `np.diff` of the scaled signal |
| `flank` | `"rising"` | which flanks become triggers: `"rising"`, `"falling"` or `"both"`. Rising-only matches the PicoHarp marker channel in the PTU files. Legacy files, whose trigger stream drops at laser-on, need `"both"` |
| `reference` | `None` | optional averaged `Signal`; when given, trigger indices/times are mapped onto its (binned) time axis |
| `slope_threshold` | `0.1` | fraction of the max `|diff|` used to qualify a slope sample as a flank (only used by `method="slope"`) |

A single detected flank gives a one-point list. Passing `reference=kpfm` is important when correlating with an averaged KPFM
signal, so the trigger `index` values line up with the binned axis used by
`Match`.

## Peak detection

```python
onsets = z.find_peaks(z.signal(stream=TRIGGER_STREAM, units="V"), prominence=2.0, left_drop=0.3)
```

Detects trigger onsets via `scipy.signal.find_peaks` and walks back from each
peak to the point where the signal drops below `left_drop * peak`. Returns the
onsets as a list of trigger points on the stream's axis (the same format as
`laser_triggers`), or `None` with a warning if no peak is found. Extra keywords
are forwarded to `scipy.signal.find_peaks`.
