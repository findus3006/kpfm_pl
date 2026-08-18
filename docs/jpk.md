# JPK reader (`kpfm_pl.JPK`)

Loads JPK AFM image files (`.jpk`, a TIFF-based container). Each channel
(height, surface potential, photon counter, ...) is returned as a calibrated
[`Image`](../src/kpfm_pl/core.py) with physical axes and z-units. No plotting
or colormaps live in the loader – those belong to the optional
[`plotting`](../src/kpfm_pl/plotting.py) helpers.

The runnable example is
[`examples/example_jpk.py`](../examples/example_jpk.py), which loads
`jpk_test_(Drift1).jpk` from the `Test data/` folder.

## Loading a file

```python
from kpfm_pl import JPK

jpk = JPK("overview.jpk")
print(jpk.channel_names())   # e.g. ['Height (measured)', 'Precision 5', ...]
```

The reader iterates over the TIFF pages, reads JPK's private tags, applies the
per-channel calibration (`value = raw * scale + offset`) and flips the image
vertically so row 0 is at the bottom. Channels are keyed by
`"<name> <trace>"` (a `retrace` suffix distinguishes retrace scans).
The unnamed 64x64 page is retained as `jpk.thumbnail` for reference but is not
listed as a measurement channel. Its scan dimensions and calibration metadata
are still used for the calibrated channels.

Attributes after loading:

| Attribute | Meaning |
|-----------|---------|
| `channels` | `{key: ndarray}` calibrated channel arrays |
| `units` | `{key: unit}` z-unit per channel |
| `x`, `y` | physical scan axes (metres), derived from the scan factors `dx`/`dy` |
| `thumbnail` | unnamed preview page, separate from the measurement channels |

## Channel calibration

Calibration is looked up by the channel's type tag. Known channel types:

| Channel type | Units |
|--------------|-------|
| `height`, `measuredHeight` | `m` |
| `phase` | `deg` |
| `vDeflection` | `N` |
| `photon-counter0`, `photon-counter1` | `counts/s` |
| `precision-5`, `precision-6`, `precision-7` (raw) | `V` |
| `error` | `V` |

Channels with an unknown type are returned uncalibrated with empty units. If a
known channel is missing its scale/offset tag, a `ValueError` is raised.
The cantilever sensitivity and spring constant are parsed into SI values when
the file is loaded. With processing enabled, a Precision 6 voltage image is
multiplied by these two calibration values and returned in newtons. Missing,
malformed or non-finite force calibration raises a `ValueError` instead of
returning an incorrectly calibrated image.

## Getting channels as Images

```python
afm  = jpk.image("Height (measured)", process=True)   # Image
kpfm = jpk.image("Precision 5")                        # process=True by default
all_imgs = jpk.images(["Height (measured)", "Precision 5"])  # {name: Image}
```

- `image(channel, process=True)` – return one channel. When `process=True`,
  height channels are line-levelled (`linear_line_level`) and zeroed
  (`set_zero`), and Precision 6 channels are converted from volts to newtons.
- `images(channels=None, process=True)` – return several channels as
  `{name: Image}` (all channels when `channels=None`).
- `axes()` – `{"x": ..., "y": ...}` scan axes in metres.

## Plotting an Image

`Image` provides physical extents for `imshow`:

| Property / method | Meaning |
|-------------------|---------|
| `image.extent` | `(xmin, xmax, ymin, ymax)` in the stored unit (metres) |
| `image.extent_um` | the extent rescaled to micrometres |
| `image.extent_scaled(unit)` | extent in `"m"`, `"mm"`, `"um"`/`"µm"` or `"nm"`; x/y source units are scaled independently |

```python
import matplotlib.pyplot as plt
plt.imshow(afm["data"], extent=afm.extent_um, origin="lower")
```

Or use the optional [`plotting`](../src/kpfm_pl/plotting.py) helpers, as in the
example, for Gwyddion-style colormaps, scaled colorbars and interactive ROIs:

```python
from kpfm_pl import plotting

fig, ax, im, cbar = plotting.plot_image(
    afm, cmap=plotting.cmap_gold, cbar_scale=1e9,
    cbar_label="Height (nm)", extent_unit="um", title="Topography")
circle_params = plotting.make_circle(ax)   # interactive circular ROI
ax.add_patch(Circle(circle_params[0],circle_params[1], edgecolor = "red", fill=False)) # Adds circle patch to axis object
```

`get_save_path(name="AFM")` returns a `processed/<stem> <name>.png` path
(creating the `processed/` folder) for saving figures next to the source file.
