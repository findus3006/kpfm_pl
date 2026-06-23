"""Reader for JPK AFM image files (``.jpk``, TIFF-based).

Each channel (height, surface potential, photon counter, ...) is returned as an
:class:`~kpfm_pl.core.Image` with calibrated physical axes and z-units. No
plotting or colormaps here -- those belong in the caller / an optional plotting
layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from .core import Image
from .utils import linear_line_level, set_zero

__all__ = ["JPK"]

# JPK private TIFF tag numbers (as strings, matching tifffile's tag names).
_TAG_CHANNEL = "32848"
_TAG_NAME = "32850"
_TAG_TRACE = "32851"
_TAG_NX, _TAG_NY = "ImageWidth", "ImageLength"
_TAG_DX, _TAG_DY = "32834", "32835"

# Channel type -> (scale tag, offset tag, units).
_CHANNEL_CALIBRATION = {
    "phase": ("32980", "32981", "deg"),
    "vDeflection": ("33076", "33077", "N"),
    "photon-counter1": ("32980", "32981", "counts/s"),
    "photon-counter0": ("32980", "32981", "counts/s"),
    "precision-5": ("32980", "32981", "V"),
    "precision-6": ("32980", "32981", "V"),
    "precision-7": ("32980", "32981", "V"),
    "measuredHeight": ("32980", "32981", "m"),
    "error": ("33028", "33029", "V"),
    "height": ("33076", "33077", "m"),
}


class JPK:
    """Load a JPK AFM image file.

    Parameters
    ----------
    filepath :
        Path to the ``.jpk`` file.

    Attributes
    ----------
    channels : dict[str, numpy.ndarray]
        Raw (calibrated) channel arrays keyed by ``"<name> <trace>"``.
    """

    def __init__(self, filepath: str | Path) -> None:
        self.filepath = Path(filepath)
        self.channels: dict[str, np.ndarray] = {}
        self.units: dict[str, str] = {}
        self.x = np.array([])
        self.y = np.array([])
        self.dx = 0.0
        self.dy = 0.0
        self._read()

    # ------------------------------------------------------------------ read
    def _read(self) -> None:
        with tifffile.TiffFile(self.filepath) as tif:
            for page in tif.pages:
                tags = {tag.name: tag.value for tag in page.tags.values()}
                name = tags.get(_TAG_NAME)

                trace = "retrace" if "retrace : true" in str(tags.get(_TAG_TRACE, "")) else ""
                self._update_axes(tags, name)

                calib = _CHANNEL_CALIBRATION.get(tags.get(_TAG_CHANNEL))
                image = np.flip(page.asarray().astype(float), axis=0)
                if calib is not None:
                    scale_tag, offset_tag, units = calib
                    image = image * tags[scale_tag] + tags[offset_tag]
                else:
                    units = ""

                key = f"{name} {trace}".strip()
                self.channels[key] = image
                self.units[key] = units

    def _update_axes(self, tags: dict[str, Any], name: str | None) -> None:
        nx, ny = tags.get(_TAG_NX), tags.get(_TAG_NY) 
        if name is None: # Only the thumbnail channel includes the scan factors dx and dy, whose name is None
            self.dx, self.dy = tags.get(_TAG_DX), tags.get(_TAG_DY)
        if self.dx != 0.0 and self.dy != 0.0:
            self.x = self.dx / nx * np.arange(nx)
            self.y = self.dy / ny * np.arange(ny)


    # ------------------------------------------------------------------ access
    def channel_names(self) -> list[str]:
        """List of available channel keys."""
        return list(self.channels)

    def axes(self) -> dict[str, np.ndarray]:
        """Physical scan axes as ``{"x": ..., "y": ...}`` (metres)."""
        return {"x": self.x, "y": self.y}

    def image(self, channel: str, process: bool = True) -> Image:
        """Return one ``channel`` as an :class:`Image`.

        Parameters
        ----------
        channel :
            Channel key from :meth:`channel_names`.
        process :
            When ``True`` apply default processing (line-levelling + zeroing for
            height channels).
        """
        data = self.channels[channel]
        if process:
            data = self._process(channel, data)
        return Image(data, self.x, self.y, label=channel,
                     units={"x": "m", "y": "m", "z": self.units.get(channel, "")},
                     meta={"file": self.filepath.stem})

    def images(self, channels: list[str] | None = None, process: bool = True) -> dict[str, Image]:
        """Return several channels as ``{name: Image}`` (all channels by default)."""
        names = channels if channels else self.channel_names()
        return {name: self.image(name, process) for name in names}

    @staticmethod
    def _process(channel: str, data: np.ndarray) -> np.ndarray:
        if "height" in channel.lower():
            return set_zero(linear_line_level(data))
        return data
