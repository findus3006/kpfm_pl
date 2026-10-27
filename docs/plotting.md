# Plotting and interactive helpers (`kpfm_pl.plotting`, `kpfm_pl.selector`)

Optional matplotlib helpers for figure styling, image plots and interactive
work: zoom/pan with the mouse, and drawing regions of interest (ROIs) on images
and time traces. They need matplotlib (`pip install -e .[plot]`) and are
imported lazily, so `import kpfm_pl` stays headless. Import them either from
the package or from the submodules:

```python
from kpfm_pl import plot_image, enable_zoom_pan, InteractiveBoxSelector
# or
from kpfm_pl.plotting import plot_image, enable_zoom_pan
from kpfm_pl.selector import InteractiveBoxSelector
```

The runnable examples are
[`examples/example_jpk.py`](../examples/example_jpk.py) (image plotting and a
circular ROI) and [`examples/example_flid.py`](../examples/example_flid.py)
(selecting time-trace regions and comparing their FLIDs).

## Interactive backend

Everything mouse-driven (`enable_zoom_pan`, `make_circle`, `crop_roi` and the
selector classes) needs an **interactive matplotlib backend**:

| Where | Setting | Result |
|-------|---------|--------|
| Script | default (Qt/Tk) | separate figure window |
| Jupyter / VS Code notebook | `%matplotlib qt` | separate figure window |
| Jupyter / VS Code notebook | `%matplotlib widget` (needs `pip install ipympl`) | interactive figure inside the notebook |

With `%matplotlib inline` figures are static images, so no mouse events reach
the helpers. `make_circle` and `crop_roi` block until the window is closed,
which works best with `%matplotlib qt`.

## Figure styling

### `set_scaled_rcparams`

```python
from kpfm_pl.plotting import set_scaled_rcparams

set_scaled_rcparams((16, 9))           # call before creating the figure
fig, ax = plt.subplots(figsize=(16, 9))
```

Scales font sizes, line widths and tick sizes in `plt.rcParams` to the target
figure size, so a `(16, 9)` figure looks like a default `(9, 6)` one scaled up.
`base` (default `(9, 6)`) is the reference size the defaults are tuned for. It
changes global rcParams, so it affects every figure created afterwards.

### Colormaps

| Name | Use |
|------|-----|
| `cmap_gold` | Gwyddion "Gold", for AFM height channels |
| `cmap_spring` | Gwyddion "Spring", for KPFM / surface potential |

### `plot_image` and `set_cbar_formatter`

```python
from kpfm_pl import plotting

fig, ax, im, cbar = plotting.plot_image(
    afm, cmap=plotting.cmap_gold, cbar_scale=1e9,
    cbar_label="Height (nm)", extent_unit="um", title="Topography")
```

`plot_image` shows an [`Image`](../src/kpfm_pl/core.py) with its extent in
`extent_unit` (`"m"`, `"mm"`, `"um"`, `"nm"`), adds a colorbar and returns
`(fig, ax, im, cbar)`. Pass `ax=` to draw into an existing axis; extra keywords
go to `imshow`.

`set_cbar_formatter(cbar, data, scale)` formats colorbar ticks as
`value * scale` (e.g. `1e9` for m → nm). It uses one decimal when the displayed
span is below `threshold` (default `5`), otherwise none.

## Mouse-wheel zoom and pan

```python
from kpfm_pl.plotting import enable_zoom_pan

fig, ax = plt.subplots()
ax.plot(tt["x"], tt["y"])
enable_zoom_pan(fig)        # all axes of the figure
plt.show()
```

| Input | Action |
|-------|--------|
| Scroll wheel | zoom in/out around the cursor |
| `Shift` + wheel | zoom x only |
| `Ctrl` + wheel | zoom y only |
| Middle-button drag | pan the axis under the cursor |
| `x` / `y` held while dragging | pan along one axis only |

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `target` | `None` | a `Figure` (all its axes, including ones added later), a single `Axes`, or `None` for `plt.gcf()` |
| `base_scale` | `1.2` | zoom factor per wheel step |
| `pan_button` | `2` | mouse button used for panning (1 left, 2 middle, 3 right) |

Notes:

- Log and other non-linear scales are zoomed in their own scale space, so a
  log axis zooms by decades, not linearly.
- Colorbar axes are never zoomed or panned.
- The matplotlib toolbar (rectangle zoom, pan, home) keeps working; *home*
  restores the original view.
- With `%matplotlib widget` the wheel no longer scrolls the notebook while the
  cursor is over the figure.
- The function returns the callback ids; pass each to
  `fig.canvas.mpl_disconnect` to switch zoom/pan off again.

The selectors below use only the left and right mouse buttons, so they work
together with `enable_zoom_pan`. Zoom in to place an ROI precisely, then zoom
back out.

## ROI selectors

All selectors attach to an existing axis, work in **data coordinates** and
keep running until you disconnect them or close the figure. Call
`plt.show(block=True)` after creating one so the script waits while you draw.

### `InteractiveBoxSelector`: several rectangular ROIs

```python
from kpfm_pl.selector import InteractiveBoxSelector

fig, ax = plt.subplots()
ax.plot(tt["x"], tt["y"])
selector = InteractiveBoxSelector(ax)
enable_zoom_pan(ax)
plt.show(block=True)
rois = selector.get_boxes()   # [{"x_min", "x_max", "y_min", "y_max"}, ...]
```

| Input | Action |
|-------|--------|
| Left-drag on empty area | draw a new box |
| Left-drag inside a box | move it |
| Left-drag an edge/corner handle | resize it |
| Right-click | deselect the active box |
| `Delete` | remove the active (red) box |
| `Enter` | call `on_save(boxes)` if given |

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `on_save` | `None` | callback receiving the box list when `Enter` is pressed |
| `box_color`, `box_alpha` | `"royalblue"`, `0.25` | box appearance |
| `handle_size` | `None` | handle half-width in data units; `None` = 0.5 % of the visible x-range (updated when you zoom) |

`get_boxes()` returns plain dicts ordered by drawing order, with `x_min <
x_max` and `y_min < y_max`. `disconnect()` detaches all event handlers.

### `SingleBoxSelector`: exactly one ROI

Same controls as `InteractiveBoxSelector`, but drawing a new box replaces the
old one, right-click deletes the box, and `Enter` also closes the figure.

### `DraggableCircle`: one circular ROI

```python
from kpfm_pl.selector import DraggableCircle

circle = DraggableCircle(ax)                                # draw from scratch
circle = DraggableCircle(ax, center=(1.0, 2.0), radius=0.5) # or start pre-placed
plt.show(block=True)
center, radius = circle.get_params()                        # (None, None) if none
```

Left-drag draws the circle (center at the press point, radius = drag
distance), left-drag inside moves it, right-click deletes it.

## One-call interactive helpers

These create the selector, show the figure, block until it is closed and
return the result. Both enable wheel zoom and middle-drag pan by default and
list the controls in the figure. Pass `zoom_pan=False` to turn that off.

### `make_circle`

```python
fig, ax, im, cbar = plotting.plot_image(afm, cmap=plotting.cmap_gold)
circle_params = plotting.make_circle(ax)    # [center, radius] or None
```

### `crop_roi`

```python
cropped, bounds = plotting.crop_roi(image_array, cmap="gray")
# bounds = {"x_min", "x_max", "y_min", "y_max"} in pixels, or None
```

Shows a 2-D or RGB(A) array, lets you draw one box (`SingleBoxSelector`) and
returns the array sliced to that box together with the integer pixel bounds.
Without a box it returns the original array and `None`. Confirm with `Enter`
or by closing the window.

## Workflow: compare FLIDs of time-trace regions

Boxes drawn on a time trace have the same format that
[`PTU.flid`](ptu.md#flid-map) takes as `roi`: `x_min`/`x_max` is the time
window (s) and `y_min`/`y_max` the count range of the time bins to include.
So you can draw a few regions on the trace and pass each box to `flid`:

```python
import matplotlib.pyplot as plt
from kpfm_pl import PTU
from kpfm_pl.plotting import enable_zoom_pan
from kpfm_pl.selector import InteractiveBoxSelector

bintime = 0.5
ptu = PTU("scan.ptu", dark_counts=2.25)
tt = ptu.time_trace(bintime=bintime)

fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(tt["x"], tt["y"])
selector = InteractiveBoxSelector(ax)
enable_zoom_pan(ax)
plt.show(block=True)

# Shared histogram ranges -> identical bins, so the FLIDs are comparable
ranges = {"lifetime_range": (0, 20), "intensity_range": (0, tt["y"].max() + 1)}
flids = [ptu.flid(roi=roi, bintime=bintime, n_exp=1, **ranges)
         for roi in selector.get_boxes()]

fig, axes = plt.subplots(1, len(flids), sharex=True, sharey=True, squeeze=False)
for ax, flid in zip(axes[0], flids):
    extent = (*flid.meta["lifetime_range"], *flid.meta["intensity_range"])
    ax.imshow(flid["data"], extent=extent, origin="lower", aspect="auto")
    ax.set_xlabel("Lifetime (ns)")
axes[0, 0].set_ylabel(f"Counts/{bintime * 1e3:.0f} ms")
plt.show()
```

Things to keep in mind:

- Use the **same `bintime`** for the plotted trace and for `flid`. The box's
  count range refers to the counts per bin of the plotted trace, so a
  different bin time selects different bins.
- Pass the **same `lifetime_range`/`intensity_range`** to every FLID you want
  to compare. Without them each FLID gets bins fitted to its own data. The
  maps then have different bin sizes, and on shared axes the last `imshow`
  sets the limits for all of them, cropping the others. Lifetimes outside
  `lifetime_range` are dropped, so widen it if your data extends further.
- Use the ranges from `meta` as the `imshow` extent, as above. They are the
  exact image edges (see [FLID map](ptu.md#flid-map)).
- `flid` raises `ValueError` for a box that contains no photons.
- Bins with almost no counts (e.g. a box that starts before the laser turns
  on) give meaningless lifetime fits that pile up near 0 ns and 0 counts. Raise
  the box's lower edge above the background to exclude them.
[`examples/example_flid.py`](../examples/example_flid.py) is the full version:
it skips empty boxes, falls back to a default ROI when nothing is drawn, and
plots the trace with coloured ROIs above the matching FLIDs.
