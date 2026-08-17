# kpfm_pl

A small, well-structured toolbox for loading and correlating data from
combined **KPFM**, **photoluminescence** (PicoQuant PTU / TCSPC) and **AFM**
(JPK) experiments.

The design goals:

- **One package, three file types** – PTU, Zurich `.txt` and JPK `.jpk`.
- **Consistent return format** – every loader/analysis call returns a
  plotting-friendly [`Signal`](src/kpfm_pl/core.py) or
  [`Image`](src/kpfm_pl/core.py) object that behaves like a dict
  (`s["x"], s["y"], s["label"], s["units"], s["meta"]`).
- **No plotting in the core** – data is returned ready to drop into matplotlib.
- **Flexible getters** instead of hard-coded, side-effect-heavy methods.

## Installation

```bash
pip install -e .            # core (numpy, scipy, tifffile)
pip install -e .[plot]      # + matplotlib for plotting your results
pip install -e .[dev]       # + pytest, matplotlib
```

## Package layout

```
src/kpfm_pl/
    __init__.py     public API
    core.py         Signal / Image containers (the common return format)
    ptu.py          PTU  – PicoQuant TTTR-T3 loader (time trace, lifetime, FLID)
    zurich.py       Zurich – lock-in .txt loader (KPFM, triggers)
    jpk.py          JPK    – AFM image loader (calibrated channels)
    match.py        Match  – align PL time trace with KPFM signal
    fitting.py      MultiExpFit – multi-exp decay fitting + 2-D Gaussian helpers
    database.py     build/query a searchable databank of measurement pairs
    utils.py        pure signal-processing helpers (smooth, resample, ...)
    io.py           file dialogs + measurement-pair sorting
    selector.py     optional interactive matplotlib ROI selector
    plotting.py     Plotting helper functions (cmaps, matplotlib parameter scaling, ...)
```

## Quick start

### Load and correlate a PL / KPFM pair

```python
import matplotlib.pyplot as plt
from kpfm_pl import PTU, Zurich, Match

# Photoluminescence time trace (PicoQuant .ptu)
ptu = PTU("scan.ptu", dark_counts=2.25)
tt = ptu.time_trace(bintime=0.05, laser_pulse=True)        # Signal
tt_triggers = ptu.laser_triggers()                         # from the marker channel

# KPFM lock-in stream (Zurich .txt)
z = Zurich("scan.txt")
stream_names = z.stream_names()
kpfm = z.average(stream_names[0], bintime=0.05)            # Signal
z_triggers = z.laser_triggers(stream_names[1], reference=kpfm, method="slope")

# Align on the laser triggers
m = Match(kpfm, tt,
          kpfm_triggers=z_triggers,
          timetrace_triggers=tt_triggers,
          kpfm_smoothing=20, timetrace_smoothing=1)

aligned = m.to_dict()
print("Pearson:", m.pearson())
print("Best lag (s):", m.cross_correlation()["best_lag"])

k = m.kpfm_signal()        # aligned KPFM Signal (on shared grid)
t = m.timetrace_signal()   # aligned time trace Signal (on shared grid)

fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(k["x"], k["y"], label=k["label"])
ax.plot(t["x"], t["y"], label=t["label"])
ax.axvline(k["meta"]["triggers"]["left"], color="k", linestyle="--", label="Left laser trigger")
ax.axvline(k["meta"]["triggers"]["right"], color="k", linestyle="--", label="Right laser trigger")
ax.set_ylabel("Amplitude (V)")
ax.set_xlabel("Time (s)")
ax.set_title("Correlation between KPFM signal and time trace")
ax.grid(True)
ax.legend()
plt.show()
```

See [`docs/match.md`](docs/match.md) for the full alignment, scaling and
statistics manual.

### Fit a lifetime histogram

```python
states = ptu.threshold(off=0.1, on=0.6)         # {"on": Signal, "off": Signal}
fit = ptu.fit_lifetime(states["on"], n_exp=2)   # dict: A, tau, curve, quality, ...
print(fit["tau"], fit["quality"]["r_squared"])
```

See [`docs/ptu.md`](docs/ptu.md) for the full PTU manual (time traces, FLID
maps, thresholding and lifetime fitting).

### Load an AFM image

```python
from kpfm_pl import JPK

jpk = JPK("overview.jpk")
print(jpk.channel_names())
img = jpk.image("Height (measured)", process=True)   # Image
plt.imshow(img["data"], extent=img.extent_um, origin="lower")
```

See [`docs/jpk.md`](docs/jpk.md) for the full JPK manual (channels,
calibration and processing) and [`docs/zurich.md`](docs/zurich.md) for the
Zurich lock-in reader.

## Example scripts

Runnable end-to-end scripts live in [`src/examples/`](src/examples), together
with a `Test data/` folder holding sample `.ptu`, `.txt` and `.jpk` files:

| Script | What it shows |
|--------|---------------|
| `example_correlate.py` | Load a `.ptu` + `.txt` pair and align them with `Match` |
| `example_ptu.py` | Read TCSPC data, threshold and fit a lifetime histogram |
| `example_jpk.py` | Load AFM channels and plot with the `plotting` helpers |

### Build a searchable database of measurement pairs

```python
from kpfm_pl import build_database, filter_entries, save_json

entries = build_database("Path") #Specify the path to the directory containing the measurement pairs
save_json(entries, "Path" + "\\database.json") #Save the database to a JSON file

green_pairs = filter_entries(entries, wavelength_nm=532.0) #Filter the entries by wavelength
zurich, ptu = green_pairs[0].load(dark_counts=2.25) #Load the first entry with dark count correction
```

See [`docs/database.md`](docs/database.md) for the full manual (archive layout,
parsed fields, pairing logic, querying, and editing entries).

## Return-format conventions

| Container | Keys |
|-----------|------|
| `Signal`  | `x`, `y`, `label`, `units` (`{"x", "y"}`), `meta` |
| `Image`   | `data`, `x`, `y`, `label`, `units` (`{"x", "y", "z"}`), `meta`; plus `.extent` |

Both provide `to_dict()` for serialisation and support dict-style access so they
unpack cleanly into matplotlib.

## Documentation

Detailed per-module manuals live in [`docs/`](docs):

| Manual | Covers |
|--------|--------|
| [`docs/ptu.md`](docs/ptu.md) | `PTU` – TCSPC time traces, FLID, thresholding, lifetime fitting |
| [`docs/zurich.md`](docs/zurich.md) | `Zurich` – lock-in `.txt` streams and laser triggers |
| [`docs/jpk.md`](docs/jpk.md) | `JPK` – AFM channel loading, calibration and processing |
| [`docs/match.md`](docs/match.md) | `Match` – alignment, scaling and correlation of a PL/KPFM pair |
| [`docs/database.md`](docs/database.md) | `database` – searchable databank of measurement pairs |

## Tests

```bash
pytest
```

The suite (`tests/`) covers the pure-numpy parts (`utils`, `fitting`,
`core`, `plotting`) plus the loaders (`ptu`, `zurich`, `jpk`), `match` and
`database`.
