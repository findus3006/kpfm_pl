"""Common data containers shared across the package.

Every loader and analysis routine in :mod:`kpfm_pl` returns data in one of two
light-weight, plotting-friendly formats:

* :class:`Signal` -- a 1-D trace (``x``, ``y``) with optional label/units/meta.
* :class:`Image`  -- a 2-D map (``data``) with physical ``x``/``y`` axes.

Both behave like dictionaries (``signal["x"]``) so they can be unpacked directly
into matplotlib without the caller needing to know about the class, e.g.::

    s = ptu.time_trace()
    ax.plot(s["x"], s["y"], label=s["label"])

while still offering attribute access (``signal.x``) and a plain
:meth:`~Signal.to_dict` for serialisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

__all__ = ["Signal", "Image"]


@dataclass
class Signal:
    """A 1-D trace returned by loaders and analysis helpers.

    Parameters
    ----------
    x, y :
        Equal-length 1-D arrays (e.g. time and amplitude).
    label :
        Human-readable description, suitable for a legend entry.
    units :
        Mapping with ``"x"`` and ``"y"`` unit strings, e.g. ``{"x": "s", "y": "V"}``.
    meta :
        Free-form dictionary for any extra context (triggers, fit results, ...).
    """

    x: np.ndarray
    y: np.ndarray
    label: str = ""
    units: dict[str, str] = field(default_factory=lambda: {"x": "", "y": ""})
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x)
        self.y = np.asarray(self.y)

    # -- dict-like access so callers can treat it as a plain mapping -----------
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def keys(self) -> tuple[str, ...]:
        return ("x", "y", "label", "units", "meta")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain ``dict`` copy of this signal."""
        return {
            "x": self.x,
            "y": self.y,
            "label": self.label,
            "units": dict(self.units),
            "meta": dict(self.meta),
        }

    def __len__(self) -> int:
        return int(self.y.shape[0])


@dataclass
class Image:
    """A 2-D map (AFM channel, FLID, ...) with physical axes.

    Parameters
    ----------
    data :
        2-D array of shape ``(ny, nx)``.
    x, y :
        1-D coordinate arrays for the columns and rows of ``data``.
    label :
        Human-readable description (e.g. channel name).
    units :
        Mapping with ``"x"``, ``"y"`` and ``"z"`` unit strings.
    meta :
        Free-form dictionary for extra context.
    """

    data: np.ndarray
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    label: str = ""
    units: dict[str, str] = field(default_factory=lambda: {"x": "", "y": "", "z": ""})
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        if self.x is not None:
            self.x = np.asarray(self.x)
        if self.y is not None:
            self.y = np.asarray(self.y)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def keys(self) -> tuple[str, ...]:
        return ("data", "x", "y", "label", "units", "meta")

    @property
    def extent(self) -> tuple[float, float, float, float] | None:
        """``(xmin, xmax, ymin, ymax)`` for ``imshow(..., extent=...)``."""
        if self.x is None or self.y is None:
            return None
        return (
            float(np.min(self.x)),
            float(np.max(self.x)),
            float(np.min(self.y)),
            float(np.max(self.y)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain ``dict`` copy of this image."""
        return {
            "data": self.data,
            "x": self.x,
            "y": self.y,
            "label": self.label,
            "units": dict(self.units),
            "meta": dict(self.meta),
        }


def _as_mapping(obj: "Signal | Mapping[str, Any]") -> Mapping[str, Any]:
    """Accept either a :class:`Signal` or a plain mapping and return a mapping."""
    if isinstance(obj, Signal):
        return obj.to_dict()
    return obj
