"""kpfm_pl -- a toolbox for KPFM / photoluminescence / AFM data.

Loaders return plotting-friendly :class:`~kpfm_pl.core.Signal` and
:class:`~kpfm_pl.core.Image` containers (dict-like, with ``x``/``y``/``label``/
``units``/``meta``). The :class:`~kpfm_pl.match.Match` class aligns a PL time
trace with a KPFM signal for joint statistical analysis.

Example
-------
>>> from kpfm_pl import PTU, Zurich, Match
>>> ptu = PTU("scan.ptu", dark_counts=2.55)
>>> tt = ptu.time_trace(0.05, laser_pulse=False)
>>> z = Zurich("scan.txt")
>>> kpfm = z.average(bintime=0.05)
>>> m = Match(kpfm, tt,
...           kpfm_triggers=z.laser_triggers(reference=kpfm, slope_method=True),
...           timetrace_triggers=ptu.laser_triggers())
>>> result = m.to_dict()
"""

from __future__ import annotations # This is done so python does not raise a NameError when a class is referenced before it is defined. It allows for forward references in type hints.

from .core import Image, Signal
from .database import (
    Entry,
    add_entry,
    build_database,
    filter_entries,
    load_json,
    remove_entry,
    save_csv,
    save_json,
)
from .fitting import MultiExpFit, fit_gaussian2d, gaussian2d, peak_power_density
from .io import pick_files, pick_folder, sort_measurement_files
from .jpk import JPK
from .match import Match
from .ptu import PTU
from .utils import downsample_sum
from .zurich import Zurich

__version__ = "1.2.3"

#__all__ tells the package user which classes and functions are available for import when they use the * syntax. It is a list of strings that represent the names of the public objects in the module.
__all__ = [
    "Signal",
    "Image",
    "PTU",
    "Zurich",
    "JPK",
    "Match",
    "MultiExpFit",
    "gaussian2d",
    "fit_gaussian2d",
    "peak_power_density",
    "downsample_sum",
    "pick_files",
    "pick_folder",
    "sort_measurement_files",
    "Entry",
    "build_database",
    "add_entry",
    "remove_entry",
    "filter_entries",
    "save_json",
    "load_json",
    "save_csv",
    "__version__",
]


# Optional matplotlib-dependent plotting helpers, exposed lazily so importing
# the package stays headless-friendly and free of the heavy matplotlib import.
_PLOTTING_NAMES = (
    "cmap_spring",
    "cmap_gold",
    "set_scaled_rcparams",
    "set_cbar_formatter",
    "plot_image",
    "make_circle",
    "crop_roi",
)


def __getattr__(name: str):
    # Lazily expose the optional matplotlib-dependent GUI selectors so importing
    # the package stays headless-friendly. When the user does want to use them,
    # they can do e.g. ``from kpfm_pl import InteractiveBoxSelector`` and the
    # module will be imported on demand.
    if name in ("InteractiveBoxSelector", "SingleBoxSelector", "DraggableCircle"):
        from .selector import InteractiveBoxSelector, SingleBoxSelector, DraggableCircle

        # Inject both into the package namespace so subsequent attribute lookups
        # (e.g. kpfm_pl.DraggableCircle) resolve without re-importing.
        globals()["InteractiveBoxSelector"] = InteractiveBoxSelector
        globals()["SingleBoxSelector"] = SingleBoxSelector
        globals()["DraggableCircle"] = DraggableCircle

        return globals()[name]

    if name in _PLOTTING_NAMES:
        from . import plotting

        for _n in _PLOTTING_NAMES:
            globals()[_n] = getattr(plotting, _n)
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
