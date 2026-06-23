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
from .fitting import MultiExpFit
from .io import pick_files, pick_folder, sort_measurement_files
from .jpk import JPK
from .match import Match
from .ptu import PTU
from .zurich import Zurich

__version__ = "0.1.1"

#__all__ tells the package user which classes and functions are available for import when they use the * syntax. It is a list of strings that represent the names of the public objects in the module.
__all__ = [
    "Signal",
    "Image",
    "PTU",
    "Zurich",
    "JPK",
    "Match",
    "MultiExpFit",
    "pick_files",
    "pick_folder",
    "sort_measurement_files",
    "__version__",
]


def __getattr__(name: str):
    # Lazily expose the optional matplotlib-dependent GUI selector so importing
    # the package stays headless-friendly. When then user does want to use the selector, they can do
    # from kpfm_pl import InteractiveBoxSelector and the module will be imported on demand
    if name == "InteractiveBoxSelector":
        from .selector import InteractiveBoxSelector

        return InteractiveBoxSelector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
