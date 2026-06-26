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
- **No plotting in the core** – data is returned ready to drop into matplotlib;
  you stay in control of the figures.
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
    fitting.py      MultiExpFit – robust multi-exponential decay fitting
    utils.py        pure signal-processing helpers (smooth, resample, ...)
    io.py           file dialogs + measurement-pair sorting
    selector.py     optional interactive matplotlib ROI selector
```

## Quick start

### Load and correlate a PL / KPFM pair

```python
import matplotlib.pyplot as plt
from kpfm_pl import PTU, Zurich, Match

ptu = PTU("scan.ptu", dark_counts=2.25)
tt = ptu.time_trace(bintime=0.05, laser_pulse=True)        # Signal
tt_triggers = ptu.laser_triggers()

z = Zurich("scan.txt")
stream_names = z.stream_names()
kpfm = z.average(stream_names[0], bintime=0.05)                               # Signal
z_triggers = z.laser_triggers(stream_names[1], reference=kpfm, slope_method=True)

m = Match(kpfm, tt,
          kpfm_triggers=z_triggers,
          timetrace_triggers=tt_triggers,
          kpfm_smoothing=100, timetrace_smoothing=100)

aligned = m.to_dict()
print("Pearson:", m.pearson())
print("Best lag (s):", m.cross_correlation()["best_lag"])

fig, ax = plt.subplots(figsize=(16,9))
ax.plot(t["x"], t["y"], label=t["label"])
ax.plot(k["x"], k["y"], label=k["label"])
ax.axvline(z_triggers["left"]["time"], color='k', linestyle='--', label='Left laser trigger')
ax.axvline(z_triggers["right"]["time"], color='k', linestyle='--', label='Right laser trigger')
ax.set_ylabel("Amplitude (V)")
ax.set_xlabel("Time (s)")
ax.set_title("Correlation between KPFM signal and Timetrace")
ax.grid(True)
ax.legend()
plt.show()
```

### Fit a lifetime histogram

```python
states = ptu.threshold(off=0.1, on=0.6)         # {"on": Signal, "off": Signal}
fit = ptu.fit_lifetime(states["on"], n_exp=2)   # dict with tau, A, curve, quality
print(fit["tau"], fit["quality"]["r_squared"])
```

### Load an AFM image

```python
from kpfm_pl import JPK

jpk = JPK("overview.jpk")
print(jpk.channel_names())
img = jpk.image("Height (measured)", process=True)   # Image
plt.imshow(img["data"], extent=img.extent, origin="lower")
```

## Return-format conventions

| Container | Keys |
|-----------|------|
| `Signal`  | `x`, `y`, `label`, `units` (`{"x", "y"}`), `meta` |
| `Image`   | `data`, `x`, `y`, `label`, `units` (`{"x", "y", "z"}`), `meta`; plus `.extent` |

Both provide `to_dict()` for serialisation and support dict-style access so they
unpack cleanly into matplotlib.

## Tests

```bash
pytest
```

The test suite covers the pure-numpy parts (`utils`, `fitting`) and needs no
data files.
