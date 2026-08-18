"""Reader for JPK AFM image files (``.jpk``, TIFF-based).

Each channel (height, surface potential, photon counter, ...) is returned as an
:class:`~kpfm_pl.core.Image` with calibrated physical axes and z-units. No
plotting or colormaps here -- those belong in the caller / an optional plotting
layer.
"""

from __future__ import annotations

import re
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
_TAG_META = "32791"

_SENSITIVITY = "cantilever-calibration-info.sensitivity"
_SPRING_CONSTANT = "cantilever-calibration-info.spring-constant"
_NUMBER_AND_UNIT = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*(.*?)\s*$"
)


def _is_precision_6(channel: str) -> bool:
    """Match JPK's display name (``Precision 6``) and type-style spelling."""
    return re.search(r"\bprecision[ -]6\b", channel, flags=re.IGNORECASE) is not None


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
        self.thumbnail: np.ndarray | None = None
        self.calibration: dict[str, str | float] = {}
        self._read()

    # ------------------------------------------------------------------ read
    def _read(self) -> None:
        channel_shape: tuple[int, int] | None = None
        with tifffile.TiffFile(self.filepath) as tif:
            for page in tif.pages:
                tags = {tag.name: tag.value for tag in page.tags.values()}
                name = tags.get(_TAG_NAME)
                if name is None:
                    self.thumbnail = np.flip(page.asarray(), axis=0)
                    self._read_scan_metadata(tags)
                    self._read_calibration_metadata(tags.get(_TAG_META, ""))
                    continue

                trace = (
                    "retrace"
                    if "retrace : true" in str(tags.get(_TAG_TRACE, ""))
                    else ""
                )

                calib = _CHANNEL_CALIBRATION.get(tags.get(_TAG_CHANNEL))
                image = np.flip(page.asarray().astype(float), axis=0)
                self._validate_page_shape(tags, image.shape, name)
                if channel_shape is None:
                    channel_shape = image.shape
                elif image.shape != channel_shape:
                    raise ValueError(
                        f"{self.filepath.name}: channel {name!r} has shape {image.shape}, "
                        f"but the other measurement channels have shape {channel_shape}."
                    )
                if calib is not None:
                    scale_tag, offset_tag, units = calib
                    try:
                        scale = self._finite_float(
                            tags[scale_tag], f"channel {name!r} scale"
                        )
                        offset = self._finite_float(
                            tags[offset_tag], f"channel {name!r} offset"
                        )
                    except KeyError as exc:
                        raise ValueError(
                            f"{self.filepath.name}: channel {name!r} is missing "
                            f"calibration tag {exc.args[0]!r}."
                        ) from exc
                    image = image * scale + offset
                else:
                    units = ""

                key = f"{name} {trace}".strip()
                self.channels[key] = image
                self.units[key] = units

        if channel_shape is not None:
            if self.dx <= 0.0 or self.dy <= 0.0:
                raise ValueError(
                    f"{self.filepath.name}: missing positive scan dimensions dx/dy."
                )
            ny, nx = channel_shape
            self.x = self.dx / nx * np.arange(nx)
            self.y = self.dy / ny * np.arange(ny)

    def _read_scan_metadata(self, tags: dict[str, Any]) -> None:
        """Read the physical scan size carried by the unnamed thumbnail page."""
        if _TAG_DX in tags:
            self.dx = self._finite_float(tags[_TAG_DX], "scan width")
        if _TAG_DY in tags:
            self.dy = self._finite_float(tags[_TAG_DY], "scan height")

    def _read_calibration_metadata(self, raw_calibration: str) -> None:
        """Read metadata and convert force-calibration values to SI floats."""
        for line in raw_calibration.split("\n"):
            if " : " in line:
                key, value = line.split(" : ", 1)
                self.calibration[key.strip()] = value.strip()

        sensitivity_scales = {
            "m/V": 1.0,
            "nm/V": 1e-9,
            "um/V": 1e-6,
            "µm/V": 1e-6,
        }
        conversions = {
            _SENSITIVITY: sensitivity_scales,
            _SPRING_CONSTANT: {"N/m": 1.0},
        }
        for key, unit_scales in conversions.items():
            if key not in self.calibration:
                continue
            raw_value = str(self.calibration[key])
            match = _NUMBER_AND_UNIT.fullmatch(raw_value)
            if match is None or match.group(2) not in unit_scales:
                expected = ", ".join(unit_scales)
                raise ValueError(
                    f"{self.filepath.name}: invalid {key!r} calibration {raw_value!r}; "
                    f"expected a finite number in {expected}."
                )
            value = float(match.group(1)) * unit_scales[match.group(2)]
            if not np.isfinite(value):
                raise ValueError(
                    f"{self.filepath.name}: {key!r} calibration must be finite."
                )
            self.calibration[key] = value

    def _finite_float(self, value: Any, description: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{self.filepath.name}: {description} must be numeric; got {value!r}."
            ) from exc
        if not np.isfinite(result):
            raise ValueError(f"{self.filepath.name}: {description} must be finite.")
        return result

    def _validate_page_shape(
        self, tags: dict[str, Any], shape: tuple[int, ...], name: str
    ) -> None:
        if len(shape) != 2:
            raise ValueError(
                f"{self.filepath.name}: channel {name!r} must be 2-D; got shape {shape}."
            )
        try:
            declared = (int(tags[_TAG_NY]), int(tags[_TAG_NX]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{self.filepath.name}: channel {name!r} has invalid TIFF dimensions."
            ) from exc
        if shape != declared:
            raise ValueError(
                f"{self.filepath.name}: channel {name!r} has shape {shape}, "
                f"but its TIFF tags declare {declared}."
            )
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
            height channels) and voltage-to-force conversion for Precision 6.
        """
        data = self.channels[channel]
        if process:
            data = self._process(channel, data)
        z_unit = self.units.get(channel, "")
        if process and _is_precision_6(channel):
            z_unit = "N"
        return Image(
            data,
            self.x,
            self.y,
            label=channel,
            units={"x": "m", "y": "m", "z": z_unit},
            meta={"file": self.filepath.stem},
        )

    def images(self, channels: list[str] | None = None, process: bool = True) -> dict[str, Image]:
        """Return several channels as ``{name: Image}`` (all channels by default)."""
        names = channels if channels else self.channel_names()
        return {name: self.image(name, process) for name in names}

    def _process(self, channel: str, data: np.ndarray) -> np.ndarray:
        if "height" in channel.lower():
            return set_zero(linear_line_level(data))
        if _is_precision_6(channel):
            try:
                sensitivity = float(self.calibration[_SENSITIVITY])
                spring_constant = float(self.calibration[_SPRING_CONSTANT])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{self.filepath.name}: Precision 6 processing requires valid "
                    "cantilever sensitivity and spring-constant calibration."
                ) from exc
            if not np.isfinite(sensitivity) or not np.isfinite(spring_constant):
                raise ValueError(
                    f"{self.filepath.name}: Precision 6 force calibration must be finite."
                )
            return sensitivity * spring_constant * data
        return data

    def get_save_path(self, name: str = "AFM"):
        png_name = f"{self.filepath.stem} {name}.png"  
        output_dir = self.filepath.parent/"processed"
        output_dir.mkdir(exist_ok=True)
        return output_dir/png_name
