import importlib
from pathlib import Path

import kpfm_pl


PUBLIC_MODULES = (
    "core",
    "database",
    "fitting",
    "io",
    "jpk",
    "match",
    "plotting",
    "ptu",
    "selector",
    "utils",
    "zurich",
)


def test_module_public_symbols_are_reexported_from_package():
    for module_name in PUBLIC_MODULES:
        module = importlib.import_module(f"kpfm_pl.{module_name}")
        for name in module.__all__:
            assert name in kpfm_pl.__all__
            assert getattr(kpfm_pl, name) is getattr(module, name)


def test_lazy_public_names_are_discoverable():
    names = dir(kpfm_pl)
    assert "plot_image" in names
    assert "InteractiveBoxSelector" in names


def test_version_has_one_package_source():
    from kpfm_pl._version import __version__

    assert kpfm_pl.__version__ == __version__ == "1.3.0"


def test_pep561_marker_is_shipped_in_source_package():
    package_dir = Path(kpfm_pl.__file__).parent
    assert (package_dir / "py.typed").is_file()
