"""Database workflow: table columns, filtering and the open database (no Qt).

The GUI shows a :mod:`kpfm_pl.database` entry list as a table. This module
describes the table columns, formats and sorts cell values, filters entries by
free text and by exact field values (``filter_entries`` semantics), and keeps
track of the open database file and of unsaved changes in
:class:`DatabaseDocument`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...database import (
    Entry,
    add_entry,
    build_database,
    filter_entries,
    load_json,
    remove_entry,
    save_csv,
    save_json,
)

__all__ = [
    "Column",
    "COLUMNS",
    "QUICK_FILTERS",
    "NONE_VALUE",
    "display_value",
    "sort_value",
    "matches_text",
    "distinct_values",
    "apply_filters",
    "missing_files",
    "DatabaseDocument",
]


@dataclass(frozen=True)
class Column:
    """One table column: the :class:`Entry` field it shows and how."""

    field: str
    header: str
    numeric: bool = False
    tooltip: str = ""


COLUMNS: tuple[Column, ...] = (
    Column("sample", "Sample", tooltip="Top-level sample folder."),
    Column("precipitation", "Precipitation", tooltip="Optional precipitation subfolder."),
    Column("date", "Date", tooltip="Date folder of the measurement."),
    Column("dr", "DR", tooltip="Dot rod / sample position (DRx in the file names)."),
    Column("scan", "Scan", numeric=True, tooltip="Scan number from the .txt file name."),
    Column("kpfm_mode", "KPFM mode", tooltip="heterodyne (hKPFM) or amplitude modulation (AM)."),
    Column("kpfm_amplitude_v", "Amplitude (V)", numeric=True,
           tooltip="KPFM modulation amplitude."),
    Column("kpfm_gain", "Gain", numeric=True, tooltip="AM gain (for reference only)."),
    Column("pid_bandwidth_hz", "PID bandwidth (Hz)", numeric=True,
           tooltip="PID bandwidth from the .txt file name."),
    Column("wavelength_nm", "Wavelength (nm)", numeric=True,
           tooltip="Excitation wavelength (440 nm when absent from the file names)."),
    Column("power_density_w_cm2", "Power density (W/cm²)", numeric=True,
           tooltip="Excitation power density from the .ptu file name."),
    Column("total_power_nw", "Total power (nW)", numeric=True,
           tooltip="Total excitation power (secondary, for reference)."),
    Column("measurement_time_s", "Time (s)", numeric=True,
           tooltip="Measurement duration (360 s when absent from the file name)."),
    Column("txt_path", "KPFM file (.txt)", tooltip="Zurich lock-in export."),
    Column("ptu_path", "PL file (.ptu)", tooltip="PicoQuant photon-counting file."),
)

#: Fields offered as drop-down quick filters, in display order.
QUICK_FILTERS: tuple[str, ...] = ("sample", "precipitation", "date", "dr", "kpfm_mode",
                                  "wavelength_nm", "power_density_w_cm2")

#: Quick-filter value selecting entries whose field is ``None``.
NONE_VALUE = "<none>"

_PATH_FIELDS = ("txt_path", "ptu_path")


def _column(field: str) -> Column:
    for column in COLUMNS:
        if column.field == field:
            return column
    raise KeyError(field)


def display_value(entry: Entry, field: str) -> str:
    """Text shown in the table cell: blank for ``None``, file name for paths."""
    value = getattr(entry, field)
    if value is None:
        return ""
    if field in _PATH_FIELDS:
        return Path(value).name
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def sort_value(entry: Entry, field: str) -> float | str:
    """Sort key of a cell; numeric columns sort numerically, blanks last."""
    value = getattr(entry, field)
    if _column(field).numeric:
        return float("inf") if value is None else float(value)
    return display_value(entry, field).lower()


def _search_text(entry: Entry) -> str:
    parts = [display_value(entry, column.field) for column in COLUMNS]
    parts += [str(entry.txt_path), str(entry.ptu_path)]
    return " ".join(parts).lower()


def matches_text(entry: Entry, text: str) -> bool:
    """``True`` when every whitespace-separated word of ``text`` occurs in the entry.

    All columns and the full file paths are searched, case-insensitively.
    """
    haystack = _search_text(entry)
    return all(word in haystack for word in text.lower().split())


def distinct_values(entries: Iterable[Entry], field: str) -> list[tuple[Any, str]]:
    """``(value, label)`` options for a quick filter on ``field``.

    Sorted numerically or alphabetically; :data:`NONE_VALUE` is appended when
    some entries have no value for ``field``.
    """
    values = {getattr(e, field) for e in entries}
    has_none = None in values
    values.discard(None)
    numeric = _column(field).numeric
    ordered = sorted(values, key=(float if numeric else lambda v: str(v).lower()))
    options = [(v, f"{v:g}" if isinstance(v, float) else str(v)) for v in ordered]
    if has_none:
        options.append((NONE_VALUE, "(none)"))
    return options


def apply_filters(entries: Iterable[Entry], text: str = "",
                  criteria: Mapping[str, Any] | None = None) -> list[Entry]:
    """Entries matching the free ``text`` and every ``field=value`` criterion.

    ``None`` values in ``criteria`` mean "no filter"; :data:`NONE_VALUE`
    selects entries whose field is ``None``.
    """
    active = {k: (None if v == NONE_VALUE else v)
              for k, v in (criteria or {}).items() if v is not None}
    selected = filter_entries(list(entries), **active)
    return [e for e in selected if matches_text(e, text)] if text.strip() else selected


def missing_files(entry: Entry) -> list[Path]:
    """The entry's ``.txt``/``.ptu`` files that do not exist (any more)."""
    return [Path(p) for p in (entry.txt_path, entry.ptu_path) if not Path(p).is_file()]


class DatabaseDocument:
    """The open database: its entries, file and unsaved-change state.

    Attributes
    ----------
    entries : list[Entry]
        The entries (mutated in place by :meth:`add` and :meth:`remove`).
    path : Path or None
        JSON file the database was opened from or last saved to.
    root : Path or None
        Archive folder the database was built from (used to classify added
        pairs); ``None`` when unknown.
    dirty : bool
        ``True`` when there are changes not yet saved to :attr:`path`.
    """

    def __init__(self) -> None:
        self.entries: list[Entry] = []
        self.path: Path | None = None
        self.root: Path | None = None
        self.dirty = False

    @property
    def is_empty(self) -> bool:
        return not self.entries

    def set_built(self, entries: list[Entry], root: str | Path) -> None:
        """Replace the database with freshly built ``entries`` (unsaved)."""
        self.entries = list(entries)
        self.root = Path(root)
        self.path = None
        self.dirty = True

    def build(self, root: str | Path) -> None:
        """Scan ``root`` with :func:`~kpfm_pl.database.build_database`."""
        self.set_built(build_database(root), root)

    def open(self, path: str | Path) -> None:
        """Load a JSON database saved with :meth:`save`."""
        path = Path(path)
        self.entries = load_json(path)
        self.path = path
        self.root = None
        self.dirty = False

    def save(self, path: str | Path) -> None:
        """Save all entries as JSON and remember ``path``."""
        path = Path(path)
        save_json(self.entries, path)
        self.path = path
        self.dirty = False

    @staticmethod
    def export_csv(entries: list[Entry], path: str | Path) -> None:
        """Write ``entries`` (e.g. the filtered table) as CSV for Excel."""
        save_csv(entries, path)

    def add(self, txt_path: str | Path, ptu_path: str | Path) -> Entry:
        """Add a ``.txt``/``.ptu`` pair.

        Sample, precipitation and date come from the folder structure when the
        pair lies inside :attr:`root`; otherwise the ``.txt`` file's folder name
        becomes the sample.
        """
        txt_path, ptu_path = Path(txt_path), Path(ptu_path)
        root = self.root
        if root is not None:
            try:
                txt_path.parent.relative_to(root)
            except ValueError:
                root = None
        entry = add_entry(self.entries, txt_path, ptu_path, root=root)
        self.dirty = True
        return entry

    def remove(self, entries: Iterable[Entry]) -> int:
        """Remove ``entries`` (matched by both paths); return how many were removed."""
        removed = 0
        for entry in list(entries):
            removed += len(remove_entry(self.entries, txt_path=entry.txt_path,
                                        ptu_path=entry.ptu_path))
        if removed:
            self.dirty = True
        return removed
