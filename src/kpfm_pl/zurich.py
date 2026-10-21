"""Reader for Zurich Instruments lock-in ``.txt`` exports.

Parses the multi-stream text format (KPFM surface potential, laser trigger, ...)
into per-stream :class:`~kpfm_pl.core.Signal` objects. No plotting -- return the
arrays and plot them yourself.
"""

from __future__ import annotations

import warnings
from pathlib import Path
import numpy as np

from .core import Signal
from .utils import TriggerPoint, bin_average, feature_scale, trigger_points

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
            # The exported time axis is negative and reversed; flip to forward
            # time. Referencing each sample to the first exported timestamp is
            # the robust equivalent of negate-and-reverse: it keeps every value
            # paired with its own elapsed time even when the axis carries an
            # end offset or jitter.
            t_raw = np.array([p[0] for p in block])
            t = t_raw - t_raw[0]
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
    def laser_triggers(self, stream: str = TRIGGER_STREAM, *, method: str = "threshold",
                       flank: str = "rising", reference: Signal | None = None,
                       slope_threshold: float = 0.1) -> list[TriggerPoint] | None:
        """Locate the laser triggers in a trigger ``stream``.

        By default only rising flanks count as triggers, matching the PicoHarp
        marker channel of the PTU files (see
        :meth:`~kpfm_pl.ptu.PTU.laser_triggers`).

        Parameters
        ----------
        stream :
            Trigger stream name.
        method :
            ``"threshold"`` (default) finds every contiguous window above
            ``0.1`` of the ``[0, 1]`` feature-scaled stream; its first sample
            is the rising and its last sample the falling flank. ``"slope"``
            detects flanks from the derivative of the scaled signal.
        flank :
            Which flanks to report: ``"rising"`` (default), ``"falling"`` or
            ``"both"``. Legacy files, whose trigger stream drops when the laser
            switches on, need ``"both"`` to recover the laser-on edge.
        reference :
            Optional averaged :class:`Signal`; when given, trigger indices and
            times are mapped onto its (binned) time axis.
        slope_threshold :
            Fraction of the maximum ``|diff|`` used to qualify a slope sample
            as a flank (only used by ``method="slope"``).

        Returns
        -------
        list[TriggerPoint] or None
            Every detected trigger as ``[{"index", "time"}, ...]`` in
            chronological order (see :func:`~kpfm_pl.utils.trigger_points`).
            ``None`` if no trigger is found.
        """
        if flank not in ("rising", "falling", "both"):
            raise ValueError(
                f"Unknown flank {flank!r}; expected 'rising', 'falling' or 'both'."
            )
        signs = {"rising": (1,), "falling": (-1,), "both": (1, -1)}[flank]
        value = self.streams[stream]["value"]
        time = self.streams[stream]["time"]
        scaled = feature_scale(value)

        if method == "threshold":
            on = np.where(scaled >= 0.1)[0]
            if on.size == 0:
                warnings.warn(
                    f"{self.filepath.stem}: no trigger found in {stream}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None
            if on.size < 2:
                warnings.warn(
                    f"{self.filepath.stem}: fewer than two on-samples in {stream}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None
            starts = np.r_[on[0], on[1:][np.diff(on) > 1]]
            ends = np.r_[on[:-1][np.diff(on) > 1], on[-1]]
            flanks = [(int(i), 1) for i in starts] + [(int(i), -1) for i in ends]
        elif method == "slope":
            slope = np.diff(scaled)
            max_abs = float(np.max(np.abs(slope))) if slope.size else 0.0
            if max_abs == 0 or not np.isfinite(max_abs):
                warnings.warn(
                    f"{self.filepath.stem}: no slope trigger found in {stream}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None
            thr = slope_threshold * max_abs
            # Walk the derivative and collect one (index, sign) per contiguous
            # flank, reporting the post-transition sample of the steepest step.
            flanks = []
            i, n = 0, len(slope)
            while i < n:
                sign = 1 if slope[i] > thr else -1 if slope[i] < -thr else 0
                if sign == 0:
                    i += 1
                    continue
                j = i
                while j < n and sign * slope[j] > thr:
                    j += 1
                peak = i + int(np.argmax(sign * slope[i:j]))
                flanks.append((peak + 1, sign))
                i = j
        else:
            raise ValueError(
                f"Unknown method {method!r}; expected 'threshold' or 'slope'."
            )

        all_idx = np.array(sorted(i for i, sign in flanks if sign in signs), dtype=int)
        if all_idx.size == 0:
            warnings.warn(
                f"{self.filepath.stem}: no {flank} flank found in {stream}.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        axis = time if reference is None else np.asarray(reference["x"], dtype=float)
        return trigger_points(time[all_idx], axis) or None

    def find_peaks(self, stream: Signal, prominence: float = 2.0, left_drop: float = 0.3,
                   **kwargs) -> list[TriggerPoint] | None:
        """Detect trigger onsets in a stream via :func:`scipy.signal.find_peaks`.

        From each peak the onset is found by walking back to where the signal
        drops below ``left_drop`` times the peak value. Extra keywords are
        forwarded to :func:`scipy.signal.find_peaks`.

        Returns
        -------
        list[TriggerPoint] or None
            The onsets as ``[{"index", "time"}, ...]`` on ``stream``'s axis.
            ``None`` if no peak is found.
        """
        from scipy import signal as _signal

        y = stream.y
        time = stream.x
        peaks, _ = _signal.find_peaks(y, prominence=prominence, **kwargs)
        onsets = []
        for p in peaks:
            threshold = y[p] * left_drop
            i = p
            while i > 0 and y[i] > threshold:
                i -= 1
            onsets.append(i)
        if not onsets:
            warnings.warn(
                f"{self.filepath.stem}: no peaks found in {stream.label}.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        return trigger_points(np.asarray(time, dtype=float)[onsets], time)
