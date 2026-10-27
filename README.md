# kpfm_pl

Python toolbox for combined **KPFM**, **photoluminescence (PL)** and **AFM**
measurements. It reads the three file types such a setup produces, aligns PL
and KPFM recordings in time, and provides the lifetime analysis and plotting
tools used on that data:

| File | Instrument | Reader |
|------|------------|--------|
| `.ptu` | PicoQuant TCSPC (TTTR T3): PL photon stream | `PTU` |
| `.txt` | Zurich Instruments lock-in export: KPFM signal and laser trigger | `Zurich` |
| `.jpk` | JPK AFM: height, KPFM, photon-counter images | `JPK` |

What you can do with it:

- **PL / TCSPC:** time traces, lifetime histograms, on/off thresholding,
  multi-exponential lifetime fits, lifetime-vs-time traces and FLID maps.
- **KPFM ↔ PL correlation:** align a PL time trace with a KPFM signal on
  their laser triggers and cut out matching windows.
- **AFM images:** calibrated channels with physical axes, line levelling.
- **Measurement database:** index an archive of `.txt`/`.ptu` pairs by
  sample, wavelength, power, … and reload pairs by query.
- **Plotting helpers (optional):** Gwyddion colormaps, figure-size-scaled
  styles, mouse-wheel zoom/pan, and interactive ROI selection on traces and
  images.

## Installation

Requires Python ≥ 3.10. Clone the repository and install it in editable mode:

```bash
git clone https://github.com/findus3006/kpfm_pl.git
cd kpfm_pl
pip install -e .            # core: numpy, scipy, tifffile
pip install -e .[plot]      # + matplotlib, for the plotting and interactive helpers
pip install -e .[gui]       # + matplotlib and PySide6, for the graphical interface
pip install -e .[dev]       # + pytest, pytest-qt, matplotlib and PySide6, for running the tests
```

The core never imports matplotlib. The plotting helpers and ROI selectors are
loaded only when you use them, so the readers also work on headless machines.

## Graphical interface

For working without code, start the desktop GUI:

```bash
kpfm-pl-gui                 # or: python -m kpfm_pl.gui
```

Each tab is one workflow: **PL lifetime (PTU)**, **FLID**, **Correlate**
(KPFM ↔ PL), **AFM (JPK)** and **Database**. Parameters sit on the left, with
an explanation on hover, and plots on the right. Every plot can be saved as
PNG/SVG/PDF and its data exported as CSV; both carry the source files and
analysis parameters as metadata. Long fits run in the background with a
progress bar and a Cancel button.
→ [`docs/gui.md`](docs/gui.md)

## Data containers

Analysis results come back as one of two containers from
[`kpfm_pl.core`](src/kpfm_pl/core.py). Both support dict-style access and
`to_dict()`:

| Container | Contents |
|-----------|----------|
| `Signal` | `x`, `y`, `label`, `units` (`{"x", "y"}`), `meta`: time traces, histograms, KPFM signals |
| `Image` | `data`, `x`, `y`, `label`, `units` (`{"x", "y", "z"}`), `meta`, plus `extent`, `extent_um`, `extent_scaled(unit)` for `imshow`: AFM channels, FLID maps |

`meta` holds everything else a result carries, e.g. the photon count matrix of
a time trace or the triggers of an aligned signal. Fit results, trigger lists
and database entries are plain dicts, lists and dataclasses, documented in the
manuals.

## Quick start

### PL time traces and lifetimes (`PTU`)

```python
from kpfm_pl import PTU

ptu = PTU("scan.ptu", dark_counts=2.25)
tt = ptu.time_trace(bintime=0.05)              # Signal: counts per bin vs time (s)
triggers = ptu.laser_triggers()                # [{"index", "time"}, ...] from the marker channel

states = ptu.threshold(off=0.1, on=0.6)        # {"on": Signal, "off": Signal} lifetime histograms
fit = ptu.fit_lifetime(states["on"], n_exp=2)  # dict: A, tau (ns), curve, quality, ...
print(fit["tau"], fit["quality"]["r_squared"])

lt = ptu.lifetime_trace(bintime=0.5, n_exp=1)  # Signal: fitted lifetime (ns) per time bin
flid = ptu.flid(bintime=0.05, n_exp=1)         # Image: occurrence over lifetime (ns) x counts
# both accept progress=lambda done, total: ... to report (or cancel) the per-bin fits
# For n_exp > 1 both use the intensity-weighted mean lifetime sum(A τ²) / sum(A τ).
```

`laser_triggers` reads the laser on/off markers that current recordings store
in the PTU marker channel. Older files without markers need
`time_trace(..., laser_pulse=False)`. Continuous-wave files load, but have no
lifetime axis, so the lifetime methods raise a `ValueError`.
→ [`docs/ptu.md`](docs/ptu.md)

### Correlate KPFM and PL (`Zurich` + `Match`)

```python
from kpfm_pl import PTU, Zurich, Match

ptu = PTU("scan.ptu", dark_counts=2.25)
tt = ptu.time_trace(bintime=0.05)

z = Zurich("scan.txt")
streams = z.stream_names()                     # [KPFM stream, trigger stream, ...]
kpfm = z.average(streams[0], bintime=0.05)     # Signal, bin-averaged
z_triggers = z.laser_triggers(streams[1], reference=kpfm, method="slope")

m = Match(kpfm, tt, kpfm_triggers=z_triggers,
          timetrace_triggers=ptu.laser_triggers(),
          kpfm_smoothing=20, timetrace_smoothing=1)
window = m.extract(m.triggers[0], m.triggers[-1])   # or m.extract(10.0, 300.0)
kp, t = window["kpfm"], window["timetrace"]         # Signals on one shared time axis
```

`Match` shifts the time trace so the first triggers coincide and resamples
both signals onto a shared grid. It does not scale amplitudes or compute
statistics; that is up to you.
→ [`docs/match.md`](docs/match.md), [`docs/zurich.md`](docs/zurich.md)

### AFM images (`JPK`)

```python
import matplotlib.pyplot as plt
from kpfm_pl import JPK, plotting

jpk = JPK("overview.jpk")
print(jpk.channel_names())
afm = jpk.image("Height (measured)", process=True)   # Image, line-levelled, in m

plt.imshow(afm["data"], extent=afm.extent_um)   # keep the default origin="upper"
# or, with colormap and scaled colorbar:
fig, ax, im, cbar = plotting.plot_image(afm, cmap=plotting.cmap_gold, cbar_scale=1e9,
                                        cbar_label="Height (nm)")
```

→ [`docs/jpk.md`](docs/jpk.md)

### Interactive plots: zoom, pan and region selection

Needs an interactive matplotlib backend: a normal script window, or
`%matplotlib qt` / `%matplotlib widget` (`ipympl`) in a notebook. Inline
notebook plots are static images.

```python
import matplotlib.pyplot as plt
from kpfm_pl.plotting import set_scaled_rcparams, enable_zoom_pan
from kpfm_pl.selector import InteractiveBoxSelector

set_scaled_rcparams((16, 9))                 # scale fonts/lines to the figure size
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(tt["x"], tt["y"])
selector = InteractiveBoxSelector(ax)        # left-drag draws boxes
enable_zoom_pan(fig)                         # wheel zoom, middle-drag pan
plt.show(block=True)

# Each box {"x_min", "x_max", "y_min", "y_max"} is a valid PTU ROI;
# shared histogram ranges make the FLIDs directly comparable
flids = [ptu.flid(roi=roi, bintime=0.05, lifetime_range=(0, 20),
                  intensity_range=(0, tt["y"].max() + 1))
         for roi in selector.get_boxes()]
```

`plotting.make_circle(ax)` (circular ROI) and `plotting.crop_roi(image)`
(crop an array to a drawn box) wrap this into one blocking call.
→ [`docs/plotting.md`](docs/plotting.md)

### Measurement database

```python
from kpfm_pl import build_database, filter_entries, save_json, load_json

entries = build_database(r"D:\Measurements")          # scans <sample>/[<precipitation>/]<date>/ folders
save_json(entries, r"D:\Measurements\database.json")  # reload later with load_json

green = filter_entries(entries, wavelength_nm=532.0, sample="CF501")
zurich, ptu = green[0].load(dark_counts=2.25)         # (Zurich, PTU) for that pair
```

→ [`docs/database.md`](docs/database.md)

### Picking files

```python
from kpfm_pl import pick_files, pick_folder

files = pick_files(r"D:\Measurements", sort=True)   # file dialog; pairs sorted by wavelength, power
jpks = pick_folder(r"D:\Measurements")               # all .jpk files below a chosen folder
```

## Package overview

| Module | Contents | Manual |
|--------|----------|--------|
| [`ptu`](src/kpfm_pl/ptu.py) | `PTU`: time traces, lifetime histograms and fits, lifetime traces, FLID, markers | [ptu.md](docs/ptu.md) |
| [`zurich`](src/kpfm_pl/zurich.py) | `Zurich`: lock-in streams, bin averaging, laser triggers, peak detection | [zurich.md](docs/zurich.md) |
| [`jpk`](src/kpfm_pl/jpk.py) | `JPK`: calibrated AFM channels, line levelling, force conversion | [jpk.md](docs/jpk.md) |
| [`match`](src/kpfm_pl/match.py) | `Match`: trigger alignment and shared-grid resampling of KPFM and PL | [match.md](docs/match.md) |
| [`database`](src/kpfm_pl/database.py) | `build_database`, `filter_entries`, `add_entry`/`remove_entry`, JSON/CSV export | [database.md](docs/database.md) |
| [`plotting`](src/kpfm_pl/plotting.py) | colormaps, `set_scaled_rcparams`, `plot_image`, `enable_zoom_pan`, `make_circle`, `crop_roi` | [plotting.md](docs/plotting.md) |
| [`selector`](src/kpfm_pl/selector.py) | `InteractiveBoxSelector`, `SingleBoxSelector`, `DraggableCircle` | [plotting.md](docs/plotting.md) |
| [`fitting`](src/kpfm_pl/fitting.py) | `MultiExpFit` (used by `PTU`); `gaussian2d`, `fit_gaussian2d`, `peak_power_density` for laser spot profiles | docstrings |
| [`io`](src/kpfm_pl/io.py) | `pick_files`, `pick_folder` (tkinter dialogs), `sort_measurement_files` | docstrings |
| [`utils`](src/kpfm_pl/utils.py) | smoothing, resampling, fit-window selection, AFM levelling, trigger points | docstrings |
| [`core`](src/kpfm_pl/core.py) | `Signal`, `Image` | – |
| [`gui`](src/kpfm_pl/gui) | optional PySide6 desktop GUI (`kpfm-pl-gui`); not imported by the core | [gui.md](docs/gui.md) |

Everything public is importable directly from `kpfm_pl`, e.g.
`from kpfm_pl import PTU, plot_image, InteractiveBoxSelector`.

## Examples

Runnable scripts in [`examples/`](examples) use the sample files in
`examples/Test data/`:

| Script | What it shows |
|--------|---------------|
| [`example_ptu.py`](examples/example_ptu.py) | Time trace with laser markers, thresholding and a two-exponential lifetime fit |
| [`example_flid.py`](examples/example_flid.py) | Draw regions on a time trace and compare their FLIDs (interactive) |
| [`example_correlate.py`](examples/example_correlate.py) | Align a marker-based `.ptu` with a Zurich `.txt` using `Match` |
| [`example_correlate_legacy.py`](examples/example_correlate_legacy.py) | The same for a legacy `.ptu` without markers (`laser_pulse=False`) |
| [`example_jpk.py`](examples/example_jpk.py) | AFM channels with the plotting helpers and an interactive circular ROI |

## Tests

```bash
pip install -e .[dev]
pytest
```

The suite in [`tests/`](tests) covers all modules, using the files in
`examples/Test data/` and synthetic data. The interactive helpers are tested
headless with simulated mouse events, and the GUI runs offscreen under
`pytest-qt`.

## License

GPL-3.0, see [`LICENSE`](LICENSE).
