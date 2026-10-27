"""JPK workflow: channel selection, display styles and a circular ROI (no Qt).

The tab plots several AFM channels side by side with ``imshow``'s default
``origin="upper"`` (the JPK reader already flips the images so they show the
right way up). A circle drawn on one channel is stored in metres, so it stays
valid when the display unit changes, and the statistics inside it are
computed on the pixel centres exactly as ``imshow`` places them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ...core import Image
from ..export import ExportPayload, build_metadata
from .files import LoadedFile

__all__ = [
    "COLORMAPS",
    "EXTENT_UNITS",
    "ChannelStyle",
    "CircleROI",
    "default_channels",
    "default_style",
    "unit_options",
    "resolve_cmap",
    "load_images",
    "pixel_centres",
    "circle_mask",
    "circle_stats",
    "image_payload",
    "stats_text",
]

#: Colormaps offered in the GUI as ``(name, display text)``.
COLORMAPS = (
    ("gold", "Gwyddion gold (height)"),
    ("spring", "Gwyddion spring (KPFM)"),
    ("Reds", "Reds (photon counts)"),
    ("viridis", "Viridis"),
    ("inferno", "Inferno"),
    ("magma", "Magma"),
    ("cividis", "Cividis"),
    ("gray", "Grey"),
    ("RdBu_r", "Red-blue (diverging)"),
)

#: Axis units offered for the image extent as ``(unit, display text)``.
EXTENT_UNITS = (("um", "µm"), ("nm", "nm"), ("mm", "mm"))
_EXTENT_SCALE = {"m": 1.0, "mm": 1e3, "um": 1e6, "nm": 1e9}
_EXTENT_LABEL = dict(EXTENT_UNITS)

# Colorbar units per SI unit as (label, factor applied to the data); the first
# entry is the default.
_UNIT_OPTIONS = {
    "m": (("nm", 1e9), ("pm", 1e12), ("µm", 1e6), ("m", 1.0)),
    "V": (("mV", 1e3), ("V", 1.0), ("µV", 1e6)),
    "N": (("nN", 1e9), ("pN", 1e12), ("N", 1.0)),
    "counts/s": (("counts/s", 1.0), ("kcounts/s", 1e-3)),
    "deg": (("deg", 1.0),),
}


@dataclass
class ChannelStyle:
    """How one channel is displayed: colormap and colorbar unit scaling."""

    cmap: str
    unit: str
    scale: float


@dataclass
class CircleROI:
    """Circular region of interest in metres (scan coordinates)."""

    x: float
    y: float
    radius: float

    def in_unit(self, unit: str) -> tuple[tuple[float, float], float]:
        """``((cx, cy), radius)`` in the display ``unit`` (``"um"``, ``"nm"``, ...)."""
        f = _EXTENT_SCALE[unit]
        return (self.x * f, self.y * f), self.radius * f

    @classmethod
    def from_unit(cls, center: Sequence[float], radius: float, unit: str) -> CircleROI:
        f = _EXTENT_SCALE[unit]
        return cls(float(center[0]) / f, float(center[1]) / f, float(radius) / f)


def default_channels(names: Sequence[str]) -> list[str]:
    """Pick height, KPFM (Precision 5) and photon-counter trace channels if present."""
    wanted = ("Height (measured)", "Precision 5", "Photon Counter 1")
    chosen = [name for name in wanted if name in names]
    return chosen or list(names[:1])


def unit_options(unit: str) -> list[tuple[str, float]]:
    """Colorbar unit choices ``(label, factor)`` for an SI ``unit``; default first."""
    return list(_UNIT_OPTIONS.get(unit, ((unit or "arb. units", 1.0),)))


def default_style(channel: str, unit: str) -> ChannelStyle:
    """Gold for height, spring for KPFM/potential, Reds for photon counters."""
    name = channel.lower()
    if "height" in name:
        cmap = "gold"
    elif "photon" in name:
        cmap = "Reds"
    elif "precision" in name or "potential" in name or "kpfm" in name:
        cmap = "spring"
    else:
        cmap = "viridis"
    label, scale = unit_options(unit)[0]
    return ChannelStyle(cmap, label, scale)


def resolve_cmap(name: str):
    """Colormap object or name accepted by ``imshow``."""
    if name in ("gold", "spring"):
        from ... import plotting

        return plotting.cmap_gold if name == "gold" else plotting.cmap_spring
    return name


def load_images(handle: LoadedFile, channels: Iterable[str],
                process: bool) -> dict[str, Image]:
    """Read ``channels`` (in the given order) as images."""
    channels = list(channels)
    if not channels:
        return {}
    with handle.lock:
        return handle.reader.images(channels, process=process)


def pixel_centres(image: Image, unit: str = "um") -> tuple[np.ndarray, np.ndarray]:
    """Display coordinates of every pixel centre, shape ``(ny, nx)`` each.

    Matches ``imshow(data, extent=image.extent_scaled(unit))`` with the default
    ``origin="upper"``: the extent spans the pixel edges, and row 0 is at the
    top (largest y).
    """
    ny, nx = image.data.shape
    x0, x1, y0, y1 = image.extent_scaled(unit)
    xc = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    yc = y1 - (np.arange(ny) + 0.5) * (y1 - y0) / ny
    return np.meshgrid(xc, yc)


def circle_mask(image: Image, circle: CircleROI, unit: str = "um") -> np.ndarray:
    """Boolean mask of the pixels whose centre lies inside ``circle``."""
    (cx, cy), r = circle.in_unit(unit)
    xx, yy = pixel_centres(image, unit)
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def circle_stats(images: Mapping[str, Image], styles: Mapping[str, ChannelStyle],
                 circle: CircleROI, unit: str = "um") -> list[dict[str, Any]]:
    """Mean/std/min/max inside ``circle`` per channel, in each channel's display unit."""
    rows = []
    for name, image in images.items():
        style = styles[name]
        values = np.asarray(image.data, dtype=float)[circle_mask(image, circle, unit)]
        values = values[np.isfinite(values)] * style.scale
        row: dict[str, Any] = {"channel": name, "unit": style.unit, "pixels": int(values.size)}
        for key, fn in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)):
            row[key] = float(fn(values)) if values.size else float("nan")
        rows.append(row)
    return rows


def stats_text(rows: Sequence[Mapping[str, Any]], circle: CircleROI | None = None,
               unit: str = "um") -> str:
    """Tab-separated statistics table (pastes into Excel/Origin)."""
    lines = []
    if circle is not None:
        (cx, cy), r = circle.in_unit(unit)
        label = _EXTENT_LABEL.get(unit, unit)
        lines.append(f"circle centre ({label})\t{cx:.6g}\t{cy:.6g}\tradius ({label})\t{r:.6g}")
    lines.append("channel\tunit\tpixels\tmean\tstd\tmin\tmax")
    for row in rows:
        lines.append("\t".join([row["channel"], row["unit"], str(row["pixels"]),
                                *(f"{row[k]:.6g}" for k in ("mean", "std", "min", "max"))]))
    return "\n".join(lines)


def image_payload(handle: LoadedFile, image: Image, *, process: bool,
                  style: ChannelStyle | None = None, circle: CircleROI | None = None,
                  stats: Sequence[Mapping[str, Any]] = ()) -> ExportPayload:
    """CSV export of one channel in SI units, with scan and circle metadata."""
    reader = handle.reader
    results: dict[str, Any] = {
        "scan_size_m": [float(reader.dx), float(reader.dy)],
        "pixels": list(image.data.shape),
    }
    if circle is not None:
        results["circle_m"] = asdict(circle)
        results["circle_stats"] = [dict(row) for row in stats]
    parameters: dict[str, Any] = {"channel": image.label, "process": bool(process)}
    if style is not None:
        parameters["display"] = asdict(style)
    meta = build_metadata(
        "JPK channel", sources=[handle.path], parameters=parameters, results=results,
        note="Data are in the SI units given in the header (x/y in m); the display "
             "scaling only affects the plot.",
    )
    name = f"{handle.stem}_{image.label}".replace(" ", "_")
    return ExportPayload(meta, image=image, default_name=name)
