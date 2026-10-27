"""CSV and figure export with embedded metadata (no Qt).

Every export carries a metadata dict describing where the data came from and
how it was computed: the analysis name, the source file(s), the parameters
entered in the GUI, scalar results (e.g. fitted lifetimes) and the
``kpfm_pl`` version.

* CSV files start with ``# key: value`` comment lines, followed by one header
  row and the data. ``numpy.loadtxt(path, delimiter=",", skiprows=1)`` (after
  the comments) or ``pandas.read_csv(path, comment="#")`` read them back.
* Figures (PNG, SVG, PDF) store the same metadata as JSON in the file's own
  metadata fields (PNG text chunk, SVG description, PDF subject).
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..core import Image, Signal

__all__ = [
    "ExportPayload",
    "FIGURE_FILTERS",
    "build_metadata",
    "describe_meta",
    "write_signals_csv",
    "write_image_csv",
    "write_payload_csv",
    "save_figure",
]

# Arrays up to this length are written out in full in the metadata; longer
# ones are summarised by their shape so the header stays readable.
_MAX_LISTED = 64

#: Qt file-dialog filter string for the supported figure formats.
FIGURE_FILTERS = "PNG image (*.png);;SVG vector graphic (*.svg);;PDF document (*.pdf)"


@dataclass
class ExportPayload:
    """What a plot can export: its data plus the metadata describing it.

    Exactly one of ``signals`` or ``image`` should be set. ``default_name`` is
    the suggested file stem in the save dialog.
    """

    metadata: dict[str, Any]
    signals: list[Signal] = field(default_factory=list)
    image: Image | None = None
    default_name: str = "export"


def _jsonable(value: Any) -> Any:
    """Convert numpy/Path values (recursively) into JSON-friendly objects."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size > _MAX_LISTED:
            return f"<array shape={list(value.shape)} dtype={value.dtype}>"
        return _jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_LISTED:
            return f"<list of {len(value)} items>"
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def describe_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exportable part of a ``Signal``/``Image`` ``meta`` dict.

    Scalars and short arrays are kept; large arrays (photon matrices, per-bin
    fit results, ...) are dropped because they belong in the data, not the
    header. Trigger lists are reduced to their times.
    """
    out: dict[str, Any] = {}
    for key, value in meta.items():
        if key == "triggers" and isinstance(value, Sequence):
            out["trigger_times_s"] = [_jsonable(p["time"]) for p in value]
            continue
        if isinstance(value, np.ndarray) and value.size > _MAX_LISTED:
            continue
        if isinstance(value, Signal | Image):
            continue
        out[key] = _jsonable(value)
    return out


def build_metadata(analysis: str, *, sources: Sequence[str | Path] = (),
                   parameters: Mapping[str, Any] | None = None,
                   results: Mapping[str, Any] | None = None,
                   **extra: Any) -> dict[str, Any]:
    """Assemble the standard metadata dict written into every export.

    Parameters
    ----------
    analysis :
        Short description, e.g. ``"PTU time trace"``.
    sources :
        Paths of the files the data was computed from.
    parameters :
        Parameters as entered in the GUI.
    results :
        Scalar results worth recording (fit parameters, offsets, ...).
    **extra :
        Further top-level entries.
    """
    meta: dict[str, Any] = {
        "analysis": analysis,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "software": f"kpfm_pl {__version__}",
        "sources": [str(Path(s)) for s in sources],
    }
    if parameters:
        meta["parameters"] = _jsonable(dict(parameters))
    if results:
        meta["results"] = _jsonable(dict(results))
    meta.update({k: _jsonable(v) for k, v in extra.items()})
    return meta


def _header_lines(metadata: Mapping[str, Any]) -> list[str]:
    lines = []
    for key, value in metadata.items():
        text = value if isinstance(value, str) else json.dumps(_jsonable(value))
        lines.append(f"# {key}: {text}")
    return lines


def _column_name(signal: Signal, axis: str) -> str:
    unit = signal.units.get(axis, "")
    base = signal.label or "signal"
    return f"{base} {axis} ({unit})" if unit else f"{base} {axis}"


def _format(value: Any) -> str:
    if isinstance(value, float | np.floating):
        return "" if np.isnan(value) else repr(float(value))
    return str(value)


def write_signals_csv(path: str | Path, signals: Sequence[Signal],
                      metadata: Mapping[str, Any]) -> Path:
    """Write one or more signals as columns of a CSV file.

    Signals sharing the same ``x`` array share one x column; otherwise every
    signal gets its own x/y column pair and shorter columns are padded with
    empty cells. Each signal's own exportable ``meta`` is added to the header
    under ``signal_meta``.
    """
    if not signals:
        raise ValueError("Nothing to export.")
    path = Path(path)
    shared_x = all(
        s.x.shape == signals[0].x.shape and np.array_equal(s.x, signals[0].x)
        for s in signals[1:]
    )
    columns: list[tuple[str, np.ndarray]] = []
    if shared_x:
        x_unit = signals[0].units["x"]
        columns.append((f"x ({x_unit})" if x_unit else "x", signals[0].x))
        columns += [(_column_name(s, "y"), s.y) for s in signals]
    else:
        for s in signals:
            columns += [(_column_name(s, "x"), s.x), (_column_name(s, "y"), s.y)]

    header = dict(metadata)
    header["signal_meta"] = {s.label or f"signal {i}": describe_meta(s.meta)
                             for i, s in enumerate(signals)}
    n_rows = max(len(c) for _, c in columns)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        for line in _header_lines(header):
            fh.write(line + "\n")
        writer = csv.writer(fh)
        writer.writerow([name for name, _ in columns])
        for i in range(n_rows):
            writer.writerow([_format(c[i]) if i < len(c) else "" for _, c in columns])
    return path


def write_image_csv(path: str | Path, image: Image,
                    metadata: Mapping[str, Any]) -> Path:
    """Write an image as a matrix: first row holds x, first column holds y.

    The top-left cell describes the axes (``y (unit) \\ x (unit)``). Without
    axes, pixel indices are used.
    """
    path = Path(path)
    data = np.asarray(image.data)
    x = image.x if image.x is not None else np.arange(data.shape[1])
    y = image.y if image.y is not None else np.arange(data.shape[0])
    units = image.units
    header = dict(metadata)
    header["image"] = {"label": image.label, "units": dict(units),
                       "shape": list(data.shape), **describe_meta(image.meta)}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        for line in _header_lines(header):
            fh.write(line + "\n")
        writer = csv.writer(fh)
        corner = f"y ({units['y']}) \\ x ({units['x']}); values in {units['z'] or 'arb.'}"
        writer.writerow([corner, *(_format(v) for v in x)])
        for yv, row in zip(y, data):
            writer.writerow([_format(yv), *(_format(v) for v in row)])
    return path


def write_payload_csv(path: str | Path, payload: ExportPayload) -> Path:
    """Write an :class:`ExportPayload` with the matching CSV writer."""
    if payload.image is not None:
        return write_image_csv(path, payload.image, payload.metadata)
    return write_signals_csv(path, payload.signals, payload.metadata)


def save_figure(figure, path: str | Path, metadata: Mapping[str, Any],
                *, dpi: int = 300) -> Path:
    """Save a matplotlib ``figure`` with ``metadata`` embedded in the file.

    The format follows the suffix (``.png``, ``.svg`` or ``.pdf``).
    """
    path = Path(path)
    fmt = path.suffix.lower().lstrip(".")
    text = json.dumps(_jsonable(dict(metadata)))
    title = str(metadata.get("analysis", "kpfm_pl figure"))
    if fmt == "png":
        file_meta = {"Title": title, "Software": f"kpfm_pl {__version__}",
                     "Description": text}
    elif fmt == "svg":
        file_meta = {"Title": title, "Description": text}
    elif fmt == "pdf":
        file_meta = {"Title": title, "Subject": text,
                     "Creator": f"kpfm_pl {__version__}"}
    else:
        raise ValueError(f"Unsupported figure format {path.suffix!r}; use .png, .svg or .pdf.")
    figure.savefig(path, dpi=dpi, metadata=file_meta, format=fmt)
    return path
