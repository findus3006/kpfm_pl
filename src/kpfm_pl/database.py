"""Build a searchable database of paired KPFM (``.txt``) / PL (``.ptu``) measurements.

The measurement archive is organised as::

    <sample>/[<precipitation>/]<date>/<DRx>_..._<...>.ptu
    <sample>/[<precipitation>/]<date>/<...>_<DRx>_..._<...>.txt

A "leaf" folder is the lowest-level directory that directly contains ``.txt``
and ``.ptu`` files -- ``.ipynb`` files and any other extensions are ignored.
Each ``.txt`` file (Zurich/KPFM lock-in export) is paired with the ``.ptu``
file (PicoQuant photon-counting export) it was recorded together with, and
the pair becomes one :class:`Entry` in the database.

Filename conventions (see docstrings on the individual parsing helpers for
the accepted variants):

``.txt`` (KPFM)
    - ``DRx`` identifies the dot rod / sample position.
    - KPFM mode: heterodyne KPFM, marked ``hKPFM<amplitude>V`` (e.g.
      ``hKPFM2V`` = heterodyne KPFM with 2 V modulation amplitude), or
      amplitude modulation, marked ``AM<gain>_<amplitude>V`` or just
      ``AM<amplitude>V`` / ``AM<gain>``. The gain number is recorded for
      reference but is not physically important and can be ``None``.
    - PID bandwidth, e.g. ``25Hz``, ``100mHz`` (milli-hertz supported).

``.ptu`` (photon counting)
    - Power density, marked e.g. ``10W`` (interpreted as 10 W/cm^2). The
      *total* power, marked e.g. ``600nW``, is recorded but not the primary
      quantity of interest.
    - Wavelength, marked e.g. ``440nm``. Defaults to 440 nm when absent.
    - Measurement time, marked e.g. ``360s``. Defaults to 360 s when absent.

Example
-------
>>> from kpfm_pl.database import build_database, save_json, load_json
>>> entries = build_database(r"C:\\Users\\rolfs\\Desktop\\Claude correlate")
>>> save_json(entries, r"C:\\Users\\rolfs\\Desktop\\Claude correlate\\database.json")
>>> entries = load_json(r"C:\\Users\\rolfs\\Desktop\\Claude correlate\\database.json")
>>> entries[0].ptu_path, entries[0].txt_path
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = [
    "Entry",
    "build_database",
    "add_entry",
    "remove_entry",
    "filter_entries",
    "save_json",
    "load_json",
    "save_csv",
]

# --------------------------------------------------------------------------- regexes
_DR_RE = re.compile(r"DR(\d+)", re.I)
_SCAN_RE = re.compile(r"Scan(\d+)", re.I)

_HETERODYNE_RE = re.compile(r"hKPFM(\d+(?:\.\d+)?)V", re.I)
_AM_GAIN_AMP_RE = re.compile(r"AM(\d+)_(\d+(?:\.\d+)?)V", re.I)
_AM_AMP_ONLY_RE = re.compile(r"AM(\d+(?:\.\d+)?)V", re.I)
_AM_GAIN_ONLY_RE = re.compile(r"AM(\d+)(?![\d_.]*V)", re.I)

_HZ_RE = re.compile(r"(\d+(?:\.\d+)?)_?\s?(m)?Hz", re.I)
_NM_RE = re.compile(r"(\d+(?:\.\d+)?)nm", re.I)
_NW_RE = re.compile(r"(\d+(?:\.\d+)?)nW", re.I)
_WCM2_RE = re.compile(r"(?<![nN])(\d+(?:\.\d+)?)W(?![a-zA-Z])")
_TIME_RE = re.compile(r"(\d+(?:\.\d+)?)s(?=[_.])", re.I)

_DATE_FULL_RE = re.compile(r"^\d{1,2}[.\-]\d{1,2}[.\-]\d{2,4}$")
_DATE_SEARCH_RE = re.compile(r"\d{1,2}[.\-]\d{1,2}[.\-]\d{2,4}")

_DEFAULT_WAVELENGTH_NM = 440.0
_DEFAULT_TIME_S = 360.0

_DATA_SUFFIXES = {".txt", ".ptu"}


# --------------------------------------------------------------------------- data model
@dataclass
class Entry:
    """One paired KPFM/PL measurement."""

    sample: str
    precipitation: str | None
    date: str | None
    dr: str | None
    scan: int | None

    kpfm_mode: str | None          # "heterodyne" or "AM"
    kpfm_amplitude_v: float | None
    kpfm_gain: int | None
    pid_bandwidth_hz: float | None

    wavelength_nm: float
    power_density_w_cm2: float | None
    total_power_nw: float | None
    measurement_time_s: float

    txt_path: Path
    ptu_path: Path

    def to_dict(self) -> dict:
        d = asdict(self)
        d["txt_path"] = str(self.txt_path)
        d["ptu_path"] = str(self.ptu_path)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Entry":
        d = dict(d)
        d["txt_path"] = Path(d["txt_path"])
        d["ptu_path"] = Path(d["ptu_path"])
        return cls(**d)

    def load(self):
        """Load the underlying :class:`~kpfm_pl.zurich.Zurich` and :class:`~kpfm_pl.ptu.PTU` objects."""
        from .ptu import PTU
        from .zurich import Zurich

        return Zurich(self.txt_path), PTU(self.ptu_path)


# --------------------------------------------------------------------------- filename parsing
def _extract_dr(name: str) -> str | None:
    m = _DR_RE.search(name)
    return f"DR{m.group(1)}" if m else None


def _extract_scan(name: str) -> int | None:
    m = _SCAN_RE.search(name)
    return int(m.group(1)) if m else None


def _extract_kpfm_mode(name: str) -> tuple[str | None, float | None, int | None]:
    """Return ``(mode, amplitude_v, gain)`` parsed from a KPFM ``.txt`` filename."""
    m = _HETERODYNE_RE.search(name)
    if m:
        return "heterodyne", float(m.group(1)), None

    m = _AM_GAIN_AMP_RE.search(name)
    if m:
        return "AM", float(m.group(2)), int(m.group(1))

    m = _AM_AMP_ONLY_RE.search(name)
    if m:
        return "AM", float(m.group(1)), None

    m = _AM_GAIN_ONLY_RE.search(name)
    if m:
        return "AM", None, int(m.group(1))

    return None, None, None


def _extract_hz(name: str) -> float | None:
    m = _HZ_RE.search(name)
    if not m:
        return None
    value = float(m.group(1))
    if m.group(2):  # "m" prefix -> milli-hertz
        value *= 1e-3
    return value


def _extract_wavelength_nm(name: str) -> float | None:
    m = _NM_RE.search(name)
    return float(m.group(1)) if m else None


def _extract_total_power_nw(name: str) -> float | None:
    m = _NW_RE.search(name)
    return float(m.group(1)) if m else None


def _extract_power_density(name: str) -> float | None:
    m = _WCM2_RE.search(name)
    return float(m.group(1)) if m else None


def _extract_time_s(name: str) -> float | None:
    m = _TIME_RE.search(name)
    return float(m.group(1)) if m else None


def parse_txt(path: Path) -> dict:
    """Parse a KPFM ``.txt`` filename into its measurement metadata."""
    stem = path.stem
    mode, amplitude, gain = _extract_kpfm_mode(stem)
    return {
        "dr": _extract_dr(stem),
        "scan": _extract_scan(stem),
        "kpfm_mode": mode,
        "kpfm_amplitude_v": amplitude,
        "kpfm_gain": gain,
        "pid_bandwidth_hz": _extract_hz(stem),
        "wavelength_nm": _extract_wavelength_nm(stem),
        "total_power_nw": _extract_total_power_nw(stem),
    }


def parse_ptu(path: Path) -> dict:
    """Parse a PL ``.ptu`` filename into its measurement metadata."""
    stem = path.stem
    return {
        "dr": _extract_dr(stem),
        "wavelength_nm": _extract_wavelength_nm(stem),
        "power_density_w_cm2": _extract_power_density(stem),
        "total_power_nw": _extract_total_power_nw(stem),
        "measurement_time_s": _extract_time_s(stem),
        "freq_hz": _extract_hz(stem),
    }


# --------------------------------------------------------------------------- folder parsing
def _classify_folder(rel_parts: tuple[str, ...]) -> tuple[str, str | None, str | None, str | None]:
    """Return ``(sample, precipitation, date, dr_hint)`` from the leaf folder's path.

    ``rel_parts`` is the leaf directory's path relative to the scan root,
    e.g. ``("CF501", "4x precipitated", "06.02.2026")``.
    """
    sample_raw = rel_parts[0]
    sample = sample_raw.split()[0] if " " in sample_raw else sample_raw

    if len(rel_parts) == 1:
        # Everything (date, DR, notes) is packed into a single folder name,
        # e.g. "JV001 DR1 07.05.2026 noLaserPulse".
        date_m = _DATE_SEARCH_RE.search(sample_raw)
        date = date_m.group(0) if date_m else None
        dr_hint = _extract_dr(sample_raw)
        return sample, None, date, dr_hint

    middle = rel_parts[1:]
    date = None
    precipitation_parts = []
    for part in middle:
        if _DATE_FULL_RE.match(part) and date is None:
            date = part
        else:
            precipitation_parts.append(part)
    precipitation = " / ".join(precipitation_parts) if precipitation_parts else None
    return sample, precipitation, date, None


def _find_leaf_folders(root: Path):
    """Yield directories that directly contain at least one ``.txt`` or ``.ptu`` file."""
    for dirpath in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
        if any(f.suffix.lower() in _DATA_SUFFIXES for f in dirpath.iterdir() if f.is_file()):
            yield dirpath


# --------------------------------------------------------------------------- pairing
def _signature(dr: str | None, hz: float | None, power_nw: float | None, wavelength_nm: float | None):
    return (dr, hz, power_nw, wavelength_nm)


def _match_score(txt_sig, ptu_sig) -> int:
    score = 0
    for t, p in zip(txt_sig, ptu_sig):
        if t is not None and p is not None and t == p:
            score += 1
    return score


def _pair_leaf_files(txt_files: list[Path], ptu_files: list[Path], folder_dr_hint: str | None):
    """Greedily pair ``.txt``/``.ptu`` files within one leaf folder.

    Files are paired by maximising the number of shared, independently
    recovered numeric tokens (DR number, PID/laser frequency, total power,
    wavelength) between a given ``.txt`` and ``.ptu`` filename. This handles
    folders with a single obvious pair as well as folders holding several
    pairs distinguished only by frequency or by power/wavelength.

    Returns a list of ``(txt_path, ptu_path)`` tuples. Unmatched files (e.g.
    a stray ``.txt`` without a companion ``.ptu``) are dropped with a
    ``print`` warning.
    """
    txt_meta = {p: parse_txt(p) for p in txt_files}
    ptu_meta = {p: parse_ptu(p) for p in ptu_files}

    # Fall back to the folder-level DR hint (needed when e.g. .ptu files
    # don't carry a DR token at all, as in "JV001 DR1 .../360s_....ptu").
    all_dr = {m["dr"] for m in txt_meta.values() if m["dr"]} | {m["dr"] for m in ptu_meta.values() if m["dr"]}
    single_dr = next(iter(all_dr)) if len(all_dr) == 1 else None
    fallback_dr = folder_dr_hint or single_dr

    for m in txt_meta.values():
        if m["dr"] is None:
            m["dr"] = fallback_dr
    for m in ptu_meta.values():
        if m["dr"] is None:
            m["dr"] = fallback_dr

    remaining_ptu = list(ptu_files)
    pairs = []
    for txt_path in txt_files:
        tm = txt_meta[txt_path]
        t_sig = _signature(tm["dr"], tm["pid_bandwidth_hz"], tm["total_power_nw"], tm["wavelength_nm"])

        best_ptu, best_score = None, -1
        for ptu_path in remaining_ptu:
            pm = ptu_meta[ptu_path]
            p_sig = _signature(pm["dr"], pm["freq_hz"], pm["total_power_nw"], pm["wavelength_nm"])
            score = _match_score(t_sig, p_sig)
            if score > best_score:
                best_ptu, best_score = ptu_path, score

        if best_ptu is None:
            print(f"Warning: no .ptu match found for {txt_path.name}")
            continue
        pairs.append((txt_path, best_ptu))
        remaining_ptu.remove(best_ptu)

    for leftover in remaining_ptu:
        print(f"Warning: no .txt match found for {leftover.name}")

    return pairs


# --------------------------------------------------------------------------- public API
def _entry_from_pair(txt_path: Path, ptu_path: Path, sample: str, precipitation: str | None,
                      date: str | None, dr_hint: str | None) -> Entry:
    tm = parse_txt(txt_path)
    pm = parse_ptu(ptu_path)

    dr = tm["dr"] or pm["dr"] or dr_hint
    wavelength = pm["wavelength_nm"] or tm["wavelength_nm"] or _DEFAULT_WAVELENGTH_NM
    time_s = pm["measurement_time_s"] or _DEFAULT_TIME_S
    total_power = pm["total_power_nw"] if pm["total_power_nw"] is not None else tm["total_power_nw"]

    return Entry(
        sample=sample,
        precipitation=precipitation,
        date=date,
        dr=dr,
        scan=tm["scan"],
        kpfm_mode=tm["kpfm_mode"],
        kpfm_amplitude_v=tm["kpfm_amplitude_v"],
        kpfm_gain=tm["kpfm_gain"],
        pid_bandwidth_hz=tm["pid_bandwidth_hz"],
        wavelength_nm=wavelength,
        power_density_w_cm2=pm["power_density_w_cm2"],
        total_power_nw=total_power,
        measurement_time_s=time_s,
        txt_path=txt_path,
        ptu_path=ptu_path,
    )


def build_database(root: str | Path) -> list[Entry]:
    """Recursively scan ``root`` and build the list of paired :class:`Entry` objects."""
    root = Path(root)
    entries: list[Entry] = []

    for leaf in _find_leaf_folders(root):
        files = [f for f in leaf.iterdir() if f.is_file()]
        txt_files = sorted(f for f in files if f.suffix.lower() == ".txt")
        ptu_files = sorted(f for f in files if f.suffix.lower() == ".ptu")
        if not txt_files or not ptu_files:
            continue

        rel_parts = leaf.relative_to(root).parts
        sample, precipitation, date, dr_hint = _classify_folder(rel_parts)

        for txt_path, ptu_path in _pair_leaf_files(txt_files, ptu_files, dr_hint):
            entries.append(_entry_from_pair(txt_path, ptu_path, sample, precipitation, date, dr_hint))

    return entries


def add_entry(entries: list[Entry], txt_path: str | Path, ptu_path: str | Path,
              root: str | Path | None = None, **overrides) -> Entry:
    """Parse a new ``.txt``/``.ptu`` pair, append it to ``entries`` and return it.

    ``root`` is the archive root used to derive ``sample``/``precipitation``/
    ``date`` from the pair's folder location, exactly like :func:`build_database`
    does. If omitted, the immediate parent folder of ``txt_path`` is treated as
    the "sample" folder (no precipitation/date). Any :class:`Entry` field can
    be overridden explicitly via keyword arguments, e.g.
    ``add_entry(entries, txt, ptu, date="01.01.2027")``.

    Parameters
    ----------
    entries :
        The list to append to (mutated in place).
    txt_path, ptu_path :
        Paths to the new ``.txt``/``.ptu`` files.
    root :
        Archive root for folder classification (optional).
    **overrides :
        Explicit values that take precedence over the parsed/derived ones.

    Examples
    --------
    >>> entry = add_entry(entries, "DR4_25_Hz_hKPFM2V.txt", "DR4_360s_25Hz_10W_001.ptu",
    ...                    root=r"C:\\Users\\rolfs\\Desktop\\Claude correlate")
    """
    txt_path, ptu_path = Path(txt_path), Path(ptu_path)

    if root is not None:
        rel_parts = txt_path.parent.relative_to(Path(root)).parts
        sample, precipitation, date, dr_hint = _classify_folder(rel_parts)
    else:
        sample, precipitation, date, dr_hint = txt_path.parent.name, None, None, None

    entry = _entry_from_pair(txt_path, ptu_path, sample, precipitation, date, dr_hint)
    if overrides:
        entry = Entry(**{**entry.to_dict(), **overrides, "txt_path": txt_path, "ptu_path": ptu_path})
    entries.append(entry)
    return entry


def remove_entry(entries: list[Entry], *, txt_path: str | Path | None = None,
                  ptu_path: str | Path | None = None) -> list[Entry]:
    """Remove and return entries matching ``txt_path`` and/or ``ptu_path`` (mutates ``entries``).

    Paths must match the full ``Entry.txt_path``/``Entry.ptu_path`` (as stored
    in the database), not just the filename.

    Examples
    --------
    >>> remove_entry(entries, txt_path=entries[0].txt_path)
    >>> remove_entry(entries, ptu_path=entries[3].ptu_path)
    """
    if txt_path is None and ptu_path is None:
        raise ValueError("Provide at least one of txt_path or ptu_path to identify the entry to remove.")

    txt_path = Path(txt_path) if txt_path is not None else None
    ptu_path = Path(ptu_path) if ptu_path is not None else None

    def _matches(e: Entry) -> bool:
        return (txt_path is None or e.txt_path == txt_path) and (ptu_path is None or e.ptu_path == ptu_path)

    removed = [e for e in entries if _matches(e)]
    entries[:] = [e for e in entries if not _matches(e)]
    return removed


def filter_entries(entries: list[Entry], **criteria) -> list[Entry]:
    """Return the entries whose fields match every ``field=value`` criterion.

    Any :class:`Entry` field can be filtered on (``sample``, ``precipitation``,
    ``date``, ``dr``, ``scan``, ``kpfm_mode``, ``kpfm_amplitude_v``,
    ``kpfm_gain``, ``pid_bandwidth_hz``, ``wavelength_nm``,
    ``power_density_w_cm2``, ``total_power_nw``, ``measurement_time_s``,
    ``txt_path``, ``ptu_path``). Matching is by equality; combine multiple
    keywords for an AND filter.

    Examples
    --------
    >>> filter_entries(entries, wavelength_nm=532.0)
    >>> filter_entries(entries, sample="CF501", dr="DR2")
    """
    return [e for e in entries if all(getattr(e, key) == value for key, value in criteria.items())]


def save_json(entries: list[Entry], path: str | Path) -> None:
    """Save the database to a JSON file."""
    path = Path(path)
    path.write_text(json.dumps([e.to_dict() for e in entries], indent=2), encoding="utf-8")


def load_json(path: str | Path) -> list[Entry]:
    """Load a database previously saved with :func:`save_json`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Entry.from_dict(d) for d in data]


def save_csv(entries: list[Entry], path: str | Path) -> None:
    """Save the database to a CSV file (paths included, easy to open in Excel)."""
    path = Path(path)
    if not entries:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(entries[0].to_dict().keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in entries:
            writer.writerow(e.to_dict())
