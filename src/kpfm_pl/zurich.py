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
from .utils import (
    TriggerCollection,
    TriggerPoint,
    bin_average,
    feature_scale,
    nearest_index,
    trigger_dict,
)

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
                       reference: Signal | None = None,
                       slope_threshold: float = 0.1) -> TriggerCollection | None:
        """Locate the laser triggers in a trigger ``stream``.

        Parameters
        ----------
        stream :
            Trigger stream name.
        method :
            ``"threshold"`` (default) locates the on-pulse edges from a
            ``[0, 1]`` feature-scaled threshold. ``"slope"`` detects **all**
            rising and falling flanks of the scaled signal, keeps them in the
            ``"all"`` entry of the returned dict, and reports the first and
            last flank as ``left``/``right``.
        reference :
            Optional averaged :class:`Signal`; when given, trigger indices are
            mapped onto its (binned) time axis.
        slope_threshold :
            Fraction of the maximum ``|diff|`` used to qualify a slope sample
            as a flank (only used by ``method="slope"``).

        Returns
        -------
        dict or None
            ``{"left": {"index", "time"}, "right": {"index", "time"},
            "all": [{"index", "time"}, ...]}``. The ``"all"`` list holds
            every detected flank in the same point format as ``left`` and
            ``right``. For threshold detection, every contiguous on-window
            contributes its first and last sample.
            ``None`` if no trigger is found.
        """
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
            all_idx = np.column_stack((starts, ends)).ravel().astype(int)
            left_idx, right_idx = int(all_idx[0]), int(all_idx[-1])
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
            # Walk the derivative and collect one index per contiguous flank,
            # reporting the post-transition sample of the steepest step.
            flanks: list[int] = []
            i, n = 0, len(slope)
            while i < n:
                if slope[i] > thr:
                    j = i
                    while j < n and slope[j] > thr:
                        j += 1
                    peak = i + int(np.argmax(slope[i:j]))
                    flanks.append(peak + 1)
                    i = j
                elif slope[i] < -thr:
                    j = i
                    while j < n and slope[j] < -thr:
                        j += 1
                    peak = i + int(np.argmin(slope[i:j]))
                    flanks.append(peak + 1)
                    i = j
                else:
                    i += 1
            if len(flanks) < 2:
                warnings.warn(
                    f"{self.filepath.stem}: fewer than two flanks in {stream}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None
            all_idx = np.array(sorted(flanks), dtype=int)
            left_idx, right_idx = int(all_idx[0]), int(all_idx[-1])
        else:
            raise ValueError(
                f"Unknown method {method!r}; expected 'threshold' or 'slope'."
            )

        left_t, right_t = float(time[left_idx]), float(time[right_idx])
        all_times = time[all_idx].astype(float)
        if reference is not None:
            ref_t = np.asarray(reference["x"], dtype=float)
            left_idx = nearest_index(ref_t, left_t)
            right_idx = nearest_index(ref_t, right_t)
            left_t, right_t = float(ref_t[left_idx]), float(ref_t[right_idx])
            all_idx = np.array([nearest_index(ref_t, t) for t in all_times], dtype=int)
            all_times = ref_t[all_idx].astype(float)

        print(f"{self.filepath.stem}: trigger spacing {right_t - left_t:.2f} s")
        if left_t == right_t:
            print(f"{self.filepath.stem}: Only one trigger found. Left and right trigger coincide.")
        out = trigger_dict(left_idx, left_t, right_idx, right_t)
        points: list[TriggerPoint] = [
            {"index": int(index), "time": float(trigger_time)}
            for index, trigger_time in zip(all_idx, all_times, strict=True)
        ]
        return {**out, "all": points}

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
