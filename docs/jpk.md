# JPK AFM images (`kpfm_pl.JPK`)

Reads JPK AFM image files (`.jpk`). Each channel (height, surface
potential, photon counter, ...) is returned as a calibrated `Image` with
physical axes and units, ready for plotting; colormaps and interactive
helpers are in [plotting.md](plotting.md).

The runnable example is
[`examples/example_jpk.py`](../examples/example_jpk.py), which loads
`jpk_test_(Drift1).jpk` from the `Test data/` folder.

## Opening a file

```python
from kpfm_pl import JPK

jpk = JPK("overview.jpk")
print(jpk.channel_names())   # e.g. ['Height (measured)', 'Precision 5', ...]
```

Every channel is calibrated on opening (`value = raw × scale + offset`, with
the factors stored in the file) and turned the right way up for
matplotlib's default `imshow(..., origin="upper")`. Channels are named
`"<name> <trace>"` (a `retrace` suffix marks retrace scans). The small
preview image of the file is kept as `jpk.thumbnail` but is not a
measurement channel.

Attributes after loading:

| Attribute | Meaning |
|-----------|---------|
| `channels` | `{key: ndarray}` calibrated channel arrays |
| `units` | `{key: unit}` z-unit per channel |
| `x`, `y` | physical scan axes (metres) |
| `thumbnail` | unnamed preview page, separate from the measurement channels |

## Channel calibration

The calibration depends on the channel type. Known types:

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

## Getting a calibrated channel

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
plt.imshow(afm["data"], extent=afm.extent_um)   # default origin="upper"
```

Or use the optional [plotting helpers](plotting.md), as in the example, for
Gwyddion-style colormaps, scaled colorbars and interactive ROIs:

```python
from matplotlib.patches import Circle
from kpfm_pl import plotting

fig, ax, im, cbar = plotting.plot_image(
    afm, cmap=plotting.cmap_gold, cbar_scale=1e9,
    cbar_label="Height (nm)", extent_unit="um", title="Topography")
circle_params = plotting.make_circle(ax)   # interactive circular ROI
ax.add_patch(Circle(circle_params[0], circle_params[1], edgecolor="red", fill=False))
```

## Measuring the drift between two scans

`image_shift` estimates the translation between two images of equal shape
by cross-correlation, with sub-pixel precision. Rows point down in the stored arrays (row 0 is the top of the
scan), so convert to scan coordinates like this:

```python
from kpfm_pl import JPK, image_shift

a = JPK("scan1.jpk").image("Height (measured)")
b = JPK("scan2.jpk").image("Height (measured)")
drow, dcol, corr = image_shift(a["data"], b["data"], return_peak=True)
dx = dcol * (a.x[1] - a.x[0])     # metres, to the right
dy = -drow * (a.y[1] - a.y[0])    # metres, upwards
```

`corr` is the correlation coefficient at the peak: close to 1 for a clean
match, small values mean the shift is unreliable. The *AFM shift* tab of the
GUI does the same for up to four files, also on a window around a circle.

## Saving figures next to the data

`jpk.get_save_path(name="AFM")` returns a `processed/<stem> <name>.png`
path next to the source file (and creates the `processed/` folder).
