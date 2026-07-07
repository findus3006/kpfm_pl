from pathlib import Path

from kpfm_pl.database import add_entry, build_database, filter_entries, load_json, remove_entry, save_json


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _make_archive(root: Path) -> None:
    # CF501 / 8x precipitated -- no date subfolder, two DR pairs distinguished by DR alone.
    base = root / "CF501" / "8x precipitated"
    _touch(base / "DR2_25_Hz_hKPFM2V.txt")
    _touch(base / "DR2_360s_50Hz_550nW_15.2W_004.ptu")
    _touch(base / "DR3_25_Hz_hKPFM2V_Scan1.txt")
    _touch(base / "DR3_360s_50Hz_550nW_12.6W_005.ptu")

    # CF501 / 4x precipitated / <date> -- single pair, no wavelength in either file.
    base = root / "CF501" / "4x precipitated" / "06.02.2026"
    _touch(base / "06.02_DR1_25Hz_hKPFM2V.txt")
    _touch(base / "DR1_360s_25Hz_10W_002.ptu")

    # JS249 / <date> -- multiple pairs per DR, disambiguated by shared frequency.
    base = root / "JS249" / "02.03.2026"
    _touch(base / "JS249_02,03_DR1_100Hz_AM17_Scan1.txt")
    _touch(base / "JS249_02,03_DR1_10Hz_AM17_Scan1.txt")
    _touch(base / "JS249_02,03_DR1_1Hz_AM17_Scan1.txt")
    _touch(base / "JS249_02,03_DR1_500Hz_AM34_2V_Scan1.txt")
    _touch(base / "DR1_360s_100Hz_4.67W_Scan1_000.ptu")
    _touch(base / "DR1_360s_10Hz_4.67W_Scan1_001.ptu")
    _touch(base / "DR1_360s_1Hz_4.67W_Scan1_000.ptu")
    _touch(base / "DR1_360s_500Hz_4.67W_Scan1_000.ptu")

    # JV001 .../<date + notes> -- files directly in the sample folder, DR missing
    # from the .ptu filenames, disambiguated by shared power+wavelength.
    base = root / "JV001 DR1 07.05.2026 noLaserPulse"
    _touch(base / "JV001_DR1_25Hz_AM13_1V_1200nW_440nm_Scan1.txt")
    _touch(base / "JV001_DR1_25Hz_AM13_1V_200nW_440nm_Scan1.txt")
    _touch(base / "360s_1200nW_440nm_000.ptu")
    _touch(base / "360s_200nW_440nm_000.ptu")

    # Irrelevant notebook files that must be ignored.
    _touch(root / "correlation attempt.ipynb")


def test_build_database_pairs_and_parses(tmp_path):
    _make_archive(tmp_path)
    entries = build_database(tmp_path)

    by_key = {(e.sample, e.precipitation, e.dr, e.pid_bandwidth_hz): e for e in entries}

    # -- CF501 / 8x precipitated ------------------------------------------------
    e = by_key[("CF501", "8x precipitated", "DR2", 25.0)]
    assert e.kpfm_mode == "heterodyne"
    assert e.kpfm_amplitude_v == 2.0
    assert e.power_density_w_cm2 == 15.2
    assert e.wavelength_nm == 440.0  # default, absent from filenames
    assert e.measurement_time_s == 360.0
    assert e.ptu_path.name == "DR2_360s_50Hz_550nW_15.2W_004.ptu"

    e = by_key[("CF501", "8x precipitated", "DR3", 25.0)]
    assert e.scan == 1
    assert e.power_density_w_cm2 == 12.6

    # -- CF501 / 4x precipitated / 06.02.2026 -----------------------------------
    e = by_key[("CF501", "4x precipitated", "DR1", 25.0)]
    assert e.date == "06.02.2026"
    assert e.power_density_w_cm2 == 10.0

    # -- JS249 -- multiple pairs per DR, matched via frequency -------------------
    for hz, expected_ptu_suffix in [
        (100.0, "100Hz_4.67W_Scan1_000.ptu"),
        (10.0, "10Hz_4.67W_Scan1_001.ptu"),
        (1.0, "1Hz_4.67W_Scan1_000.ptu"),
        (500.0, "500Hz_4.67W_Scan1_000.ptu"),
    ]:
        e = by_key[("JS249", None, "DR1", hz)]
        assert e.date == "02.03.2026"
        assert e.ptu_path.name.endswith(expected_ptu_suffix)

    e = by_key[("JS249", None, "DR1", 500.0)]
    assert e.kpfm_gain == 34
    assert e.kpfm_amplitude_v == 2.0

    e = by_key[("JS249", None, "DR1", 100.0)]
    assert e.kpfm_gain == 17
    assert e.kpfm_amplitude_v is None

    # -- JV001 -- DR missing from .ptu names, matched via power+wavelength -------
    jv_entries = [e for e in entries if e.sample == "JV001"]
    assert len(jv_entries) == 2
    for e in jv_entries:
        assert e.dr == "DR1"
        assert e.date == "07.05.2026"
        assert e.wavelength_nm == 440.0
    powers = sorted(e.total_power_nw for e in jv_entries)
    assert powers == [200.0, 1200.0]
    for e in jv_entries:
        if e.total_power_nw == 1200.0:
            assert e.ptu_path.name == "360s_1200nW_440nm_000.ptu"
        else:
            assert e.ptu_path.name == "360s_200nW_440nm_000.ptu"

    # ipynb files must not create spurious leaf folders / entries
    assert all(e.sample != "correlation attempt" for e in entries)


def test_save_and_load_json_roundtrip(tmp_path):
    _make_archive(tmp_path)
    entries = build_database(tmp_path)
    json_path = tmp_path / "database.json"
    save_json(entries, json_path)
    loaded = load_json(json_path)

    assert len(loaded) == len(entries)
    original_by_paths = {(e.txt_path, e.ptu_path): e for e in entries}
    for e in loaded:
        original = original_by_paths[(e.txt_path, e.ptu_path)]
        assert e == original


def test_remove_and_add_entry_roundtrip(tmp_path):
    _make_archive(tmp_path)
    entries = build_database(tmp_path)
    n = len(entries)

    target = entries[0]
    removed = remove_entry(entries, txt_path=target.txt_path)
    assert len(entries) == n - 1
    assert removed == [target]
    assert all(e.txt_path != target.txt_path for e in entries)

    restored = add_entry(entries, target.txt_path, target.ptu_path, root=tmp_path)
    assert restored == target
    assert len(entries) == n


def test_remove_entry_requires_a_key(tmp_path):
    _make_archive(tmp_path)
    entries = build_database(tmp_path)
    try:
        remove_entry(entries)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_add_entry_supports_overrides(tmp_path):
    _make_archive(tmp_path)
    entries = build_database(tmp_path)
    target = entries[0]
    removed = remove_entry(entries, txt_path=target.txt_path)[0]

    entry = add_entry(entries, removed.txt_path, removed.ptu_path, root=tmp_path,
                       measurement_time_s=123.0)
    assert entry.measurement_time_s == 123.0
    assert entry.sample == target.sample


def test_filter_entries(tmp_path):
    _make_archive(tmp_path)
    entries = build_database(tmp_path)

    js249 = filter_entries(entries, sample="JS249")
    assert len(js249) == 4
    assert all(e.sample == "JS249" for e in js249)

    dr1_500hz = filter_entries(entries, sample="JS249", dr="DR1", pid_bandwidth_hz=500.0)
    assert len(dr1_500hz) == 1

    heterodyne = filter_entries(entries, kpfm_mode="heterodyne")
    assert len(heterodyne) == 3
    assert all(e.sample == "CF501" for e in heterodyne)

    none_found = filter_entries(entries, wavelength_nm=532.0)
    assert none_found == []  # no 532 nm data in this synthetic fixture
