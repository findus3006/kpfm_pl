"""AFM shift workflow: the same circle on several JPK scans (no Qt).

Up to :data:`MAX_FILES` ``.jpk`` files are compared. One of them is the
*reference*: the circle drawn on it is copied to the same scan position (in
metres from the scan origin) on every other file, so a feature that drifted
between scans is visibly off-centre there.

The shift is measured in two ways:

* **manually** – every file has its own circle (initially a copy of the
  reference). The user drags it onto the feature; the shift is the distance
  between that circle and the reference circle.
* **automatically** – :func:`kpfm_pl.utils.image_shift` cross-correlates each
  channel of a file with the same channel of the reference, either over the
  whole image or in a window around the reference circle.

Shifts follow the image axes: ``dx`` to the right, ``dy`` upwards (the JPK
reader flips the images so row 0 is at the top, i.e. at the largest y).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from ...core import Image
from ...utils import image_shift
from ..export import ExportPayload, build_metadata
from .files import LoadedFile
from .jpk import (
    EXTENT_UNITS,
    ChannelStyle,
    CircleROI,
    circle_stats,
    load_images,
    pixel_centres,
)

__all__ = [
    "MAX_FILES",
    "REGIONS",
    "ShiftResult",
    "file_labels",
    "common_channels",
    "load_all",
    "same_circle",
    "follow_reference",
    "shifted_circle",
    "pixel_pitch",
    "resample_onto",
    "window_slices",
    "detect_shift",
    "detect_shifts",
    "manual_shifts",
    "apply_detected",
    "shared_ranges",
    "shift_rows",
    "shift_text",
    "stats_rows",
    "stats_text",
    "export_payloads",
    "figure_info",
]

#: Maximum number of files compared at once.
MAX_FILES = 4

#: Where the automatic shift is computed, as ``(key, display text)``.
REGIONS = (("image", "Whole image"), ("circle", "Around the circle"))

_EXTENT_SCALE = {"m": 1.0, "mm": 1e3, "um": 1e6, "nm": 1e9}
_MIN_WINDOW = 8   # pixels per side needed for a meaningful correlation

Progress = Callable[[int, int], Any]


@dataclass
class ShiftResult:
    """Shift of file ``index`` relative to the reference, in metres.

    ``source`` is ``"circle"`` for the manually placed circle, otherwise the
    channel the automatic cross-correlation was computed on; ``peak`` is its
    correlation coefficient (``None`` for the manual shift).
    """

    index: int
    file: str
    source: str
    dx: float
    dy: float
    peak: float | None = None

    @property
    def distance(self) -> float:
        return float(math.hypot(self.dx, self.dy))


# ------------------------------------------------------------------ files
def file_labels(handles: Sequence[LoadedFile]) -> list[str]:
    """File names for display; repeated names get ``" (2)"``, ``" (3)"``, ..."""
    labels: list[str] = []
    for handle in handles:
        label, n = handle.name, 2
        while label in labels:
            label = f"{handle.name} ({n})"
            n += 1
        labels.append(label)
    return labels


def common_channels(handles: Sequence[LoadedFile]) -> list[str]:
    """Channels present in every file, in the order of the first file."""
    if not handles:
        return []
    names = list(handles[0].reader.channel_names())
    for handle in handles[1:]:
        present = set(handle.reader.channel_names())
        names = [name for name in names if name in present]
    return names


def load_all(handles: Sequence[LoadedFile], channels: Iterable[str], process: bool,
             progress: Progress | None = None) -> list[dict[str, Image]]:
    """Read ``channels`` of every file (one dict per file, in file order)."""
    channels = list(channels)
    out = []
    for i, handle in enumerate(handles):
        out.append(load_images(handle, channels, process))
        if progress is not None:
            progress(i + 1, len(handles))
    return out


# ---------------------------------------------------------------- circles
def same_circle(a: CircleROI | None, b: CircleROI | None) -> bool:
    """Equal up to rounding (circles read back from the plot differ in the last bits)."""
    if a is None or b is None:
        return a is b
    scale = max(abs(a.radius), abs(b.radius), 1e-12)
    return all(abs(u - v) <= 1e-9 * scale
               for u, v in ((a.x, b.x), (a.y, b.y), (a.radius, b.radius)))


def follow_reference(circles: Sequence[CircleROI | None], ref: int,
                     new: CircleROI | None) -> list[CircleROI | None]:
    """File circles after the reference circle changed to ``new``.

    Circles that were still a copy of the old reference circle (never moved)
    follow it; moved circles keep their centre but take the new radius, so
    the statistics stay comparable. Removing the reference circle removes all.
    """
    old = circles[ref]
    out: list[CircleROI | None] = []
    for i, circle in enumerate(circles):
        if i == ref or new is None:
            out.append(new if i == ref else None)
        elif circle is None or old is None or same_circle(circle, old):
            out.append(CircleROI(new.x, new.y, new.radius))
        else:
            out.append(CircleROI(circle.x, circle.y, new.radius))
    return out


def shifted_circle(circle: CircleROI, dx: float, dy: float) -> CircleROI:
    """``circle`` moved by ``(dx, dy)`` metres."""
    return CircleROI(circle.x + dx, circle.y + dy, circle.radius)


# ------------------------------------------------------- automatic shift
def pixel_pitch(image: Image) -> tuple[float, float]:
    """Pixel width and height in metres, as ``imshow`` draws them."""
    ny, nx = image.data.shape
    x0, x1, y0, y1 = image.extent_scaled("m")
    return (x1 - x0) / nx, (y1 - y0) / ny


def resample_onto(reference: Image, image: Image) -> np.ndarray:
    """``image`` interpolated onto the pixel centres of ``reference``.

    Both images are placed by their scan coordinates (metres from the scan
    origin). Pixels of ``reference`` outside ``image`` become NaN. Images with
    the same grid are returned unchanged.
    """
    data = np.asarray(image.data, dtype=float)
    same_grid = (data.shape == reference.data.shape and np.allclose(
        image.extent_scaled("m"), reference.extent_scaled("m"), rtol=1e-9, atol=1e-15))
    if same_grid:
        return data
    xx, yy = pixel_centres(reference, "m")
    ny, nx = data.shape
    x0, x1, y0, y1 = image.extent_scaled("m")
    col = (xx - x0) / ((x1 - x0) / nx) - 0.5
    row = (y1 - yy) / ((y1 - y0) / ny) - 0.5
    finite = np.isfinite(data)
    filled = np.where(finite, data, float(np.mean(data[finite])) if finite.any() else 0.0)
    return ndimage.map_coordinates(filled, [row, col], order=1, mode="constant",
                                   cval=np.nan)


def window_slices(image: Image, circle: CircleROI,
                  window: float = 3.0) -> tuple[slice, slice]:
    """Rows/columns of a square around ``circle``, ``window`` circle diameters wide.

    Raises ``ValueError`` if the window holds fewer than 8 x 8 pixels.
    """
    ny, nx = image.data.shape
    x0, _, _, y1 = image.extent_scaled("m")
    px, py = pixel_pitch(image)
    col_c = (circle.x - x0) / px - 0.5
    row_c = (y1 - circle.y) / py - 0.5
    half_c = window * circle.radius / px
    half_r = window * circle.radius / py
    c0, c1 = max(0, math.floor(col_c - half_c)), min(nx, math.ceil(col_c + half_c) + 1)
    r0, r1 = max(0, math.floor(row_c - half_r)), min(ny, math.ceil(row_c + half_r) + 1)
    if r1 - r0 < _MIN_WINDOW or c1 - c0 < _MIN_WINDOW:
        raise ValueError(
            f"The correlation window around the circle holds only {max(r1 - r0, 0)} x "
            f"{max(c1 - c0, 0)} pixels (at least {_MIN_WINDOW} x {_MIN_WINDOW} needed). "
            "Use a larger circle or window, or the whole image.")
    return slice(r0, r1), slice(c0, c1)


def detect_shift(reference: Image, image: Image, *, circle: CircleROI | None = None,
                 window: float = 3.0) -> tuple[float, float, float]:
    """``(dx, dy, peak)``: shift of ``image`` relative to ``reference`` in metres.

    ``image`` is first resampled onto the reference grid. With ``circle``,
    only a square of ``window`` circle diameters around it is correlated,
    which measures the local drift of that feature; shifts larger than about
    half the window cannot be detected then.
    """
    ref = np.asarray(reference.data, dtype=float)
    data = resample_onto(reference, image)
    if circle is not None:
        rows, cols = window_slices(reference, circle, window)
        ref, data = ref[rows, cols], data[rows, cols]
    drow, dcol, peak = image_shift(ref, data, return_peak=True)
    px, py = pixel_pitch(reference)
    # Rows point down (row 0 is the top of the scan), y points up.
    return dcol * px, -drow * py, peak


def detect_shifts(labels: Sequence[str], images: Sequence[Mapping[str, Image]], ref: int,
                  channels: Iterable[str], *, circle: CircleROI | None = None,
                  window: float = 3.0,
                  progress: Progress | None = None) -> list[ShiftResult]:
    """Automatic shift of every non-reference file, per channel."""
    channels = list(channels)
    others = [i for i in range(len(images)) if i != ref]
    total, done = len(others) * len(channels), 0
    results = []
    for i in others:
        for name in channels:
            dx, dy, peak = detect_shift(images[ref][name], images[i][name],
                                        circle=circle, window=window)
            results.append(ShiftResult(i, labels[i], name, dx, dy, peak))
            done += 1
            if progress is not None:
                progress(done, total)
    return results


def manual_shifts(labels: Sequence[str], circles: Sequence[CircleROI | None],
                  ref: int) -> list[ShiftResult]:
    """Shift of every file's circle relative to the reference circle."""
    origin = circles[ref] if 0 <= ref < len(circles) else None
    if origin is None:
        return []
    return [ShiftResult(i, labels[i], "circle", c.x - origin.x, c.y - origin.y)
            for i, c in enumerate(circles) if i != ref and c is not None]


def apply_detected(circles: Sequence[CircleROI | None], ref: int,
                   results: Iterable[ShiftResult], channel: str) -> list[CircleROI | None]:
    """Move every file circle to the reference circle plus its detected shift on ``channel``."""
    out = list(circles)
    origin = circles[ref]
    if origin is None:
        return out
    for result in results:
        if result.source == channel and result.index != ref:
            out[result.index] = shifted_circle(origin, result.dx, result.dy)
    return out


# ----------------------------------------------------------------- display
def shared_ranges(images: Sequence[Mapping[str, Image]],
                  channels: Iterable[str]) -> dict[str, tuple[float, float]]:
    """``{channel: (vmin, vmax)}`` over all files, so the maps share one colour scale."""
    out = {}
    for name in channels:
        values = [np.asarray(per_file[name].data, dtype=float)
                  for per_file in images if name in per_file]
        finite = [v[np.isfinite(v)] for v in values]
        finite = [v for v in finite if v.size]
        if finite:
            out[name] = (float(min(v.min() for v in finite)),
                         float(max(v.max() for v in finite)))
    return out


def _unit_label(unit: str) -> str:
    return dict(EXTENT_UNITS).get(unit, unit)


def shift_rows(results: Iterable[ShiftResult], unit: str = "um") -> list[dict[str, Any]]:
    """Table rows with the shift in the display ``unit``."""
    f = _EXTENT_SCALE[unit]
    return [{"file": r.file, "source": r.source, "dx": r.dx * f, "dy": r.dy * f,
             "distance": r.distance * f, "peak": r.peak} for r in results]


def shift_text(rows: Sequence[Mapping[str, Any]], unit: str = "um") -> str:
    """Tab-separated shift table (pastes into Excel/Origin)."""
    label = _unit_label(unit)
    lines = [f"file\tsource\tdx ({label})\tdy ({label})\t|d| ({label})\tcorrelation"]
    for row in rows:
        peak = "" if row["peak"] is None else f"{row['peak']:.3g}"
        lines.append("\t".join([row["file"], row["source"],
                                *(f"{row[k]:.6g}" for k in ("dx", "dy", "distance")),
                                peak]))
    return "\n".join(lines)


def stats_rows(labels: Sequence[str], images: Sequence[Mapping[str, Image]],
               styles: Mapping[str, ChannelStyle], circles: Sequence[CircleROI | None],
               unit: str = "um") -> list[dict[str, Any]]:
    """Circle statistics per file and channel, each inside that file's own circle."""
    rows = []
    for label, per_file, circle in zip(labels, images, circles):
        if circle is None or not per_file:
            continue
        for row in circle_stats(per_file, styles, circle, unit):
            rows.append({"file": label, **row})
    return rows


def stats_text(rows: Sequence[Mapping[str, Any]]) -> str:
    """Tab-separated statistics table."""
    lines = ["file\tchannel\tunit\tpixels\tmean\tstd\tmin\tmax"]
    for row in rows:
        lines.append("\t".join([row["file"], row["channel"], row["unit"], str(row["pixels"]),
                                *(f"{row[k]:.6g}" for k in ("mean", "std", "min", "max"))]))
    return "\n".join(lines)


# ------------------------------------------------------------------ export
def _parameters(labels: Sequence[str], ref: int, channels: Sequence[str], process: bool,
                region: str, window: float) -> dict[str, Any]:
    return {"files": list(labels), "reference": labels[ref] if labels else None,
            "channels": list(channels), "process": bool(process),
            "auto_shift_region": region, "window_diameters": float(window)}


def _circles_result(labels: Sequence[str],
                    circles: Sequence[CircleROI | None]) -> dict[str, Any]:
    return {label: asdict(c) for label, c in zip(labels, circles) if c is not None}


def export_payloads(handles: Sequence[LoadedFile], ref: int,
                    results: Sequence[ShiftResult], stats: Sequence[Mapping[str, Any]],
                    circles: Sequence[CircleROI | None], *, channels: Sequence[str],
                    process: bool, region: str = "image",
                    window: float = 3.0) -> list[ExportPayload]:
    """CSV payloads: the shift table (metres) and, with a circle, its statistics."""
    labels = file_labels(handles)
    stem = f"{handles[ref].stem}_shift" if handles else "shift"
    params = _parameters(labels, ref, channels, process, region, window)
    sources = [h.path for h in handles]
    circles_m = _circles_result(labels, circles)
    payloads = []
    if results:
        table = [{"file": r.file, "source": r.source, "dx_m": r.dx, "dy_m": r.dy,
                  "distance_m": r.distance,
                  "correlation": float("nan") if r.peak is None else r.peak}
                 for r in results]
        meta = build_metadata(
            "JPK shift between scans", sources=sources, parameters=params,
            results={"circles_m": circles_m},
            note="Shift of each file relative to the reference in metres (x to the right, "
                 "y up). source 'circle' = manually placed circle, otherwise the channel "
                 "used for the automatic cross-correlation (correlation = peak "
                 "correlation coefficient).")
        payloads.append(ExportPayload(meta, table=table, default_name=f"{stem}_shifts",
                                      part="shifts"))
    if stats:
        meta = build_metadata(
            "JPK circle statistics per scan", sources=sources, parameters=params,
            results={"circles_m": circles_m},
            note="Statistics inside each file's own circle, in the display units given "
                 "in the 'unit' column.")
        payloads.append(ExportPayload(meta, table=[dict(row) for row in stats],
                                      default_name=f"{stem}_circle_stats",
                                      part="circle_stats"))
    return payloads


def figure_info(handles: Sequence[LoadedFile], ref: int, results: Sequence[ShiftResult],
                circles: Sequence[CircleROI | None], *, channels: Sequence[str],
                process: bool, region: str = "image",
                window: float = 3.0) -> tuple[str, dict[str, Any]]:
    """``(file stem, metadata)`` for saving the shift figure."""
    labels = file_labels(handles)
    meta = build_metadata(
        "JPK shift between scans", sources=[h.path for h in handles],
        parameters=_parameters(labels, ref, channels, process, region, window),
        results={"circles_m": _circles_result(labels, circles),
                 "shifts_m": [{"file": r.file, "source": r.source, "dx": r.dx,
                               "dy": r.dy, "correlation": r.peak} for r in results]})
    stem = f"{handles[ref].stem}_shift" if handles else "shift"
    return stem, meta
