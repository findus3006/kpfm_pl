"""File-discovery helpers (dialogs and measurement-pair sorting).

These are convenience wrappers around ``tkinter`` file dialogs plus a sorter that
groups measurement files by wavelength and power so a ``.ptu``/``.txt`` pair sits
next to each other. ``tkinter`` is imported lazily so importing :mod:`kpfm_pl`
stays headless-friendly.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

__all__ = ["pick_files", "pick_folder", "sort_measurement_files"]

_MEASUREMENT_RE = re.compile(r"(?P<power>\d+)nW_(?P<wavelength>\d+)nm")


def _open_dialog(title: str, filetypes, initialdir, multiple: bool = True):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if multiple:
            return filedialog.askopenfilenames(title=title, filetypes=filetypes,
                                               initialdir=initialdir)
        return filedialog.askdirectory(title=title, initialdir=initialdir)
    finally:
        root.destroy()


def pick_files(folderpath: str | Path, *, patterns=(".ptu", ".txt", ".jpk"),
               sort: bool = False, as_path: bool = True) -> tuple:
    """Open a dialog to select data files.

    Parameters
    ----------
    folderpath :
        Initial directory.
    patterns :
        Allowed file extensions (used to build the dialog filter).
    sort :
        When ``True`` group/sort the result with :func:`sort_measurement_files`.
    as_path :
        Return :class:`pathlib.Path` objects (``True``) or strings (``False``).
    """
    folder = Path(folderpath)
    if not folder.is_dir():
        raise ValueError(f"{folderpath} is not a valid directory")

    pattern_str = " ".join(f"*{p}" for p in patterns)
    selected = _open_dialog(
        "Select data files",
        [("Data files", pattern_str), ("All files", "*.*")],
        str(folder),
    )
    print(f"Selected files: {[Path(f).stem for f in selected]}")

    if sort:
        selected = sort_measurement_files(selected, as_path=as_path)
        return selected
    return tuple(Path(f) for f in selected) if as_path else tuple(selected)


def pick_folder(folderpath: str | Path, *, suffix: str = ".jpk", recursive: bool = True,
                as_path: bool = True) -> tuple:
    """Open a dialog to select a folder and return matching files within it."""
    folder = Path(folderpath)
    if not folder.is_dir():
        raise ValueError(f"{folderpath} is not a valid directory")

    selected = _open_dialog("Select a folder", None, str(folder), multiple=False)
    if not selected:
        print("No folder selected.")
        return ()

    files = list(Path(selected).rglob(f"*{suffix}") if recursive
                 else Path(selected).glob(f"*{suffix}"))
    print(f"Found files: {[f.stem for f in files]}")
    return tuple(files) if as_path else tuple(str(f) for f in files)


def sort_measurement_files(paths, *, as_path: bool = True) -> tuple:
    """Sort files by wavelength (outer) then power (inner), keeping pairs adjacent.

    Filenames are expected to contain ``<power>nW_<wavelength>nm`` (e.g.
    ``..._200nW_440nm_...``). Files without that pattern are ignored.
    """
    grouped: dict[tuple[int, int], dict[str, Path]] = defaultdict(dict) # A defaultdict(dict) automatically creates an empty dictionary when a new key is accessed. A normal dict would raise a KeyError.
    for p in paths:
        path = Path(p)
        m = _MEASUREMENT_RE.search(path.stem)
        if m:
            key = (int(m.group("wavelength")), int(m.group("power")))
            grouped[key][path.suffix] = path

    result: list[Path] = []
    for wavelength in sorted({k[0] for k in grouped}):
        for power in sorted(k[1] for k in grouped if k[0] == wavelength):
            pair = grouped[(wavelength, power)]
            result.extend(pair[suffix] for suffix in sorted(pair))

    print(f"Sorted files: {[f.stem for f in result]}")
    return tuple(result) if as_path else tuple(str(f) for f in result)
