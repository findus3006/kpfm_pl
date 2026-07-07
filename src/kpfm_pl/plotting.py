"""Optional plotting helpers (require matplotlib).

Matplotlib is a heavy dependency, so this module is **not** imported when the
package is imported. Pull it in only when you actually plot, either via the
package-level lazy attributes::

    from kpfm_pl import cmap_gold, plot_image, make_circle

or by importing the submodule explicitly::

    from kpfm_pl import plotting

Contents
--------
cmap_spring, cmap_gold
    Gwyddion-style colormaps for KPFM and height/AFM channels.
set_scaled_rcparams
    Scale matplotlib font/line rcParams to a target figure size.
set_cbar_formatter
    Attach a unit-scaling tick formatter to a colorbar.
plot_image
    One-call ``imshow`` of an :class:`~kpfm_pl.core.Image` with scaled extent,
    colorbar and formatter.
make_circle
    Interactive helper to draw a circular ROI on an axis (uses the lazily
    imported :class:`~kpfm_pl.selector.DraggableCircle`).
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

__all__ = [
    "cmap_spring",
    "cmap_gold",
    "set_scaled_rcparams",
    "set_cbar_formatter",
    "plot_image",
    "make_circle",
    "crop_roi",
]


# Gwyddion "Spring" colormap (good for KPFM / surface potential).
cmap_spring = LinearSegmentedColormap.from_list("gwyddion_spring", [
    (0.0,  (0.0,      0.0,      0.0)),
    (0.25, (0.059669, 0.380392, 0.293608)),
    (0.5,  (0.084395, 0.65098,  0.025529)),
    (0.75, (0.758756, 0.85098,  0.560646)),
    (1.0,  (1.0,      1.0,      1.0)),
])

# Gwyddion "Gold" colormap (good for AFM height channels).
cmap_gold = LinearSegmentedColormap.from_list("gwyddion_gold", [
    (0.0,      (0.0,      0.0,      0.0)),
    (0.333333, (0.345098, 0.109804, 0.0)),
    (0.666667, (0.737255, 0.501961, 0.0)),
    (1.0,      (0.988235, 0.988235, 0.501961)),
])


def set_scaled_rcparams(figsize, base=(9, 6)) -> None:
    """Update matplotlib rcParams scaled to the given figure size.

    Parameters
    ----------
    figsize :
        Target ``(width, height)`` in inches.
    base :
        Reference figure size the default sizes are tuned for.
    """
    import matplotlib.pyplot as plt

    scale = (figsize[0] / base[0] + figsize[1] / base[1]) / 2
    plt.rcParams.update({
        "axes.titlesize": 12 * scale,
        "axes.labelsize": 10 * scale,
        "xtick.labelsize": 8 * scale,
        "ytick.labelsize": 8 * scale,
        "legend.fontsize": 9 * scale,
        "figure.titlesize": 14 * scale,
        "lines.linewidth": 1.5 * scale,
        "xtick.major.size": 3 * scale,
        "ytick.major.size": 3 * scale,
        "xtick.major.width": 0.8 * scale,
        "ytick.major.width": 0.8 * scale,
    })


def set_cbar_formatter(cbar, data, scale, threshold=5.0,
                       small_fmt="{:.1f}", large_fmt="{:.0f}") -> None:
    """Format colorbar ticks as ``value * scale`` with adaptive precision.

    Parameters
    ----------
    cbar :
        The colorbar to format.
    data :
        Array (or :class:`~kpfm_pl.core.Image`/mapping) the colorbar maps; used to
        decide the tick precision from the displayed value span.
    scale :
        Multiplier applied to tick values (e.g. ``1e9`` for m -> nm).
    threshold :
        If the displayed span (``(max - min) * scale``) is below this, use
        ``small_fmt`` (more decimals), otherwise ``large_fmt``.
    small_fmt, large_fmt :
        ``str.format`` templates for the scaled tick value.
    """
    arr = np.asarray(data["data"] if hasattr(data, "keys") else data, dtype=float)
    span = (np.nanmax(arr) - np.nanmin(arr)) * scale
    fmt = small_fmt if span < threshold else large_fmt
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: fmt.format(x * scale)))


def plot_image(image, *, cmap=None, cbar_scale=1.0, cbar_label=None,
               extent_unit="um", title=None, ax=None, figsize=(10, 7), **imshow_kw):
    """Plot an :class:`~kpfm_pl.core.Image` with scaled extent and a colorbar.

    Parameters
    ----------
    image :
        An :class:`~kpfm_pl.core.Image` (provides ``data``, ``extent_scaled`` and
        ``units``).
    cmap :
        Colormap passed to ``imshow``.
    cbar_scale :
        Multiplier for the colorbar tick values (see :func:`set_cbar_formatter`),
        e.g. ``1e9`` for m -> nm.
    cbar_label :
        Colorbar label; defaults to ``image.units["z"]``.
    extent_unit :
        Length unit for the axes extent (default micrometres).
    title :
        Axis title; defaults to ``image.label``.
    ax :
        Existing axis to draw into; a new figure/axis is created when ``None``.
    figsize :
        Figure size used when ``ax`` is ``None``.
    **imshow_kw :
        Extra keyword arguments forwarded to ``ax.imshow``.

    Returns
    -------
    (fig, ax, im, cbar)
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(image["data"], extent=image.extent_scaled(extent_unit),
                   cmap=cmap, **imshow_kw)
    cbar = fig.colorbar(im)
    set_cbar_formatter(cbar, image, cbar_scale)
    cbar.set_label(cbar_label if cbar_label is not None else image["units"].get("z", ""))

    ax.set_title(title if title is not None else image["label"])
    ax.set_xlabel(f"x ({extent_unit})")
    ax.set_ylabel(f"y ({extent_unit})")
    return fig, ax, im, cbar


def make_circle(ax):
    """Interactively draw a circular ROI on ``ax`` and return ``[center, radius]``.

    Blocks until the figure window is closed. Returns ``None`` if no circle was
    drawn.
    """
    import matplotlib.pyplot as plt

    from .selector import DraggableCircle

    fig = ax.figure
    fig.suptitle("Left-click drag: draw | Drag interior: move | Right-click: delete \n Close figure window when done", fontsize=16, y=0.98)
    # fig.text(0.5, 0.02, "Close figure window when done", ha="center", va="bottom", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    draggable = DraggableCircle(ax)
    plt.show(block=True)

    center, radius = draggable.get_params()
    if center is None:
        print("No circle drawn. Processing without circle.")
        return None
    plt.close(fig)
    return [center, radius]


def crop_roi(image, *, cmap=None, title=None, figsize=(10, 7), ax=None):
    """Interactively draw a rectangular ROI on ``image`` and return the crop.

    Left-click drag on an empty area draws the box, dragging its interior
    moves it, and dragging one of the edge/corner handles resizes it.
    Right-click deletes the current box so a new one can be drawn. Confirm
    the selection by pressing ``Enter`` or simply closing the figure window
    (both behave identically). Blocks until the figure is closed.

    Parameters
    ----------
    image :
        2-D or 3-D (e.g. RGB) array to display and crop.
    cmap :
        Colormap forwarded to ``imshow`` (ignored for RGB(A) images).
    title :
        Optional axis title.
    figsize :
        Figure size used when *ax* is ``None``.
    ax :
        Existing axis to draw into; a new figure/axis is created when
        ``None``.

    Returns
    -------
    (cropped, bounds) :
        ``cropped`` is ``image`` sliced to the selected pixel bounds
        (unchanged if no ROI was drawn). ``bounds`` is a dict
        ``{"x_min", "x_max", "y_min", "y_max"}`` in pixel coordinates, or
        ``None`` if no ROI was drawn.
    """
    import matplotlib.pyplot as plt

    from .selector import SingleBoxSelector

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.imshow(image, cmap=cmap)
    if title:
        ax.set_title(title)

    instructions = (
        "Controls\n"
        "  Drag empty area    draw ROI\n"
        "  Drag box interior  move ROI\n"
        "  Drag edge/corner   resize ROI\n"
        "  Right-click        delete ROI\n"
        "  Enter / close win  confirm crop"
    )
    fig.text(0.99, 0.5, instructions, transform=fig.transFigure, fontsize=9,
              va="center", ha="right", fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", edgecolor="gray"))
    fig.subplots_adjust(right=0.78)

    selector = SingleBoxSelector(ax)
    plt.show(block=True)

    boxes = selector.get_boxes()
    if not boxes:
        print("No ROI drawn. Returning original image.")
        return image, None

    b = boxes[0]
    h, w = image.shape[:2]
    x0 = int(round(max(b["x_min"], 0)))
    x1 = int(round(min(b["x_max"], w)))
    y0 = int(round(max(b["y_min"], 0)))
    y1 = int(round(min(b["y_max"], h)))
    bounds = {"x_min": x0, "x_max": x1, "y_min": y0, "y_max": y1}
    return image[y0:y1, x0:x1, ...], bounds
