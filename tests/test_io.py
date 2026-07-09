from pathlib import Path

import pytest

from kpfm_pl.io import pick_files, pick_folder, sort_measurement_files


def test_sort_orders_by_wavelength_then_power():
    paths = [
        "meas_800nW_532nm.txt",
        "meas_200nW_440nm.ptu",
        "meas_400nW_532nm.ptu",
    ]
    out = sort_measurement_files(paths, as_path=False)
    # 440 nm before 532 nm; within 532 nm, 400 nW before 800 nW
    assert [Path(p).name for p in out] == [
        "meas_200nW_440nm.ptu",
        "meas_400nW_532nm.ptu",
        "meas_800nW_532nm.txt",
    ]


def test_sort_keeps_ptu_txt_pair_adjacent():
    paths = [
        "a_400nW_532nm.txt",
        "a_400nW_532nm.ptu",
    ]
    out = sort_measurement_files(paths, as_path=False)
    # same wavelength/power pair stays adjacent, ordered by suffix
    assert [Path(p).suffix for p in out] == [".ptu", ".txt"]


def test_sort_ignores_unmatched_filenames():
    out = sort_measurement_files(["random_file.ptu", "notes.txt"])
    assert out == ()


def test_sort_returns_path_objects_by_default():
    out = sort_measurement_files(["x_100nW_440nm.ptu"])
    assert all(isinstance(p, Path) for p in out)


def test_pick_files_invalid_dir_raises():
    with pytest.raises(ValueError, match="not a valid directory"):
        pick_files("/no/such/directory/here")


def test_pick_folder_invalid_dir_raises():
    with pytest.raises(ValueError, match="not a valid directory"):
        pick_folder("/no/such/directory/here")
