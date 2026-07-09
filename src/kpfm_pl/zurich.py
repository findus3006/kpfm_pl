"""Reader for Zurich Instruments lock-in ``.txt`` exports.

Parses the multi-stream text format (KPFM surface potential, laser trigger, ...)
into per-stream :class:`~kpfm_pl.core.Signal` objects. No plotting -- return the
arrays and plot them yourself.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .core import Signal
from .utils import bin_average, feature_scale

__all__ = ["Zurich"]

# Default stream names used in the lab setup.
KPFM_STREAM = "Demod 6 Sample Auxiliary Input 1"
TRIGGER_STREAM = "Demod 6 Sample Auxiliary Input 2"


class Zurich:
    """Load a Zurich Instruments ``.txt`` export.

    The file is parsed into ``self.streams``: a ``{name: {"time", "value"}}``
    mapping. Use :meth:`signal` to obtain a plotting-ready :class:`Signal`.

    Parameters
    ----------
    filepath :
        Path to the ``.txt`` file.
    """

    def __init__(self, filepath: str | Path) -> None:
        self.filepath = Path(filepath)
        self.streams = self._parse(self.filepath.read_text().splitlines())
        self.time = self.streams[next(iter(self.streams))]["time"]
        print(f"{self.filepath.stem}: {len(self.time)} points, "
              f"{len(self.streams)} streams: {list(self.streams)}")

    # ------------------------------------------------------------------ parse
    @staticmethod
    def _parse(lines: list[str]) -> dict[str, dict[str, np.ndarray]]:
        stream_names: list[str] = []
        data_rows: list[str] = []
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("%"):
                if "," in line and not line.startswith(("% Time", "% Module", "% Device")):
                    stream_names.append(line[1:].split(",")[0].strip())
            else:
                data_rows.append(line)

        if not stream_names:
            raise ValueError("No data streams found in header.")

        parsed = []
        skipped = 0
        for row in data_rows:
            parts = row.split(";")
            if len(parts) == 2:
                parsed.append((float(parts[0]), float(parts[1])))
            else:
                skipped += 1
        if not parsed:
            raise ValueError("No valid 'time;value' data rows found in file.")
        if skipped:
            warnings.warn(
                f"Skipped {skipped} malformed data row(s) while parsing.",
                RuntimeWarning,
                stacklevel=2,
            )
        if len(parsed) % len(stream_names) != 0:
            raise ValueError(
                f"{len(parsed)} rows not divisible by {len(stream_names)} streams."
            )

        chunk = len(parsed) // len(stream_names)
        streams: dict[str, dict[str, np.ndarray]] = {}
        for i, name in enumerate(stream_names):
            block = parsed[i * chunk:(i + 1) * chunk]
            # The exported time axis is negative and reversed; flip to forward time.
            t = np.array([-p[0] for p in block])[::-1]
            v = np.array([p[1] for p in block])
            streams[name] = {"time": t, "value": v}
        return streams

    # ------------------------------------------------------------------ access
    def stream_names(self) -> list[str]:
        """Names of all parsed streams."""
        return list(self.streams)

    def signal(self, stream: str = KPFM_STREAM, units: str = "V") -> Signal:
        """Return a stream as a :class:`Signal` (defaults to the KPFM channel)."""
        s = self.streams[stream]
        return Signal(s["time"], s["value"], label=f"{self.filepath.stem} {stream}",
                      units={"x": "s", "y": units}, meta={"stream": stream})

    def average(self, stream: str = KPFM_STREAM, bintime: float = 50e-3,
                units: str = "V") -> Signal:
        """Time-average a stream into ``bintime`` bins as a :class:`Signal`.

        The bin size is derived from the data's own sampling interval. Trailing,
        partially filled bins are averaged over their valid samples.
        """
        s = self.streams[stream]
        dt = s["time"][1] - s["time"][0]
        bin_size = max(int(round(bintime / dt)), 1)
        xb, yb = bin_average(s["time"], s["value"], bin_size)
        return Signal(xb, yb, label=f"{self.filepath.stem} {stream} (avg)",
                      units={"x": "s", "y": units},
                      meta={"stream": stream, "bintime": bintime, "averaged": True})

    # ------------------------------------------------------------------ triggers
    def laser_triggers(self, stream: str = TRIGGER_STREAM, *, slope_method: bool = False,
                       reference: Signal | None = None) -> dict[str, dict[str, float]] | None:
        """Locate the two laser triggers in a trigger ``stream``.

        Parameters
        ----------
        stream :
            Trigger stream name.
        slope_method :
            When ``True`` locate triggers via the steepest rising/falling slopes
            instead of thresholding.
        reference :
            Optional averaged :class:`Signal`; when given, trigger indices are
            mapped onto its (binned) time axis.

        Returns
        -------
        dict or None
            ``{"left": {"index", "time"}, "right": {"index", "time"}}``.
        """
        value = self.streams[stream]["value"]
        time = self.streams[stream]["time"]
        scaled = feature_scale(value)
        
        if slope_method:
            slope = np.diff(scaled)
            left_idx = int(np.argmin(slope))
            right_idx = int(np.argmax(slope))
            if left_idx > right_idx:
                left_idx, right_idx = right_idx, left_idx
        else:
            on = np.where(scaled >= 0.1)[0]
            if on.size == 0:
                warnings.warn(
                    f"{self.filepath.stem}: no trigger found in {stream}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None
            half = on.size // 2
            left_idx = int(np.max(on[:half])) if half else int(on[0])
            right_idx = int(np.max(on[half:]))

        left_t, right_t = float(time[left_idx]), float(time[right_idx])
        if reference is not None:
            ref_t = np.asarray(reference["x"])
            left_idx = int(np.argmin(np.abs(ref_t - left_t)))
            right_idx = int(np.argmin(np.abs(ref_t - right_t)))
            left_t, right_t = float(ref_t[left_idx]), float(ref_t[right_idx])

        print(f"{self.filepath.stem}: trigger spacing {right_t - left_t:.2f} s")
        return {"left": {"index": left_idx, "time": left_t},
                "right": {"index": right_idx, "time": right_t}}

    def find_peaks(self, stream: str, prominence: float = 2.0, left_drop: float = 0.3,
                   **kwargs) -> np.ndarray:
        """Detect trigger onsets in a stream via :func:`scipy.signal.find_peaks`.

        Returns an ``(N, 2)`` array of ``[index, time]`` onset points.
        """
        from scipy import signal as _signal

        y = self.streams[stream]["value"]
        time = self.streams[stream]["time"]
        peaks, _ = _signal.find_peaks(y, prominence=prominence, **kwargs)
        onsets = []
        for p in peaks:
            threshold = y[p] * left_drop
            i = p
            while i > 0 and y[i] > threshold:
                i -= 1
            onsets.append((i, time[i]))
        return np.array(onsets) if onsets else np.empty((0, 2))
