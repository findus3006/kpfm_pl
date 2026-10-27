"""Tests of the GUI database model (Qt-free) and the Database tab (offscreen)."""

from pathlib import Path

import pytest

from kpfm_pl.database import build_database, load_json
from kpfm_pl.gui.models import database as model


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _make_archive(root: Path) -> None:
    """Small archive with empty files; the database only parses file names."""
    base = root / "CF501" / "8x precipitated"
    _touch(base / "DR2_25_Hz_hKPFM2V.txt")
    _touch(base / "DR2_360s_50Hz_550nW_15.2W_004.ptu")
    _touch(base / "DR3_25_Hz_hKPFM2V_Scan1.txt")
    _touch(base / "DR3_360s_50Hz_550nW_12.6W_005.ptu")
    base = root / "JS249" / "02.03.2026"
    _touch(base / "JS249_02,03_DR1_100Hz_AM17_Scan1.txt")
    _touch(base / "JS249_02,03_DR1_10Hz_AM17_532nm_Scan1.txt")
    _touch(base / "DR1_360s_100Hz_4.67W_Scan1_000.ptu")
    _touch(base / "DR1_360s_10Hz_4.67W_532nm_Scan1_001.ptu")


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / "archive"
    _make_archive(root)
    return root


@pytest.fixture
def entries(archive):
    return build_database(archive)


# ------------------------------------------------------------------ model
def test_display_and_sort_values(entries):
    e = next(e for e in entries if e.sample == "CF501" and e.dr == "DR2")
    assert model.display_value(e, "precipitation") == "8x precipitated"
    assert model.display_value(e, "scan") == "1"
    assert model.display_value(e, "power_density_w_cm2") == "15.2"
    assert model.display_value(e, "ptu_path") == "DR2_360s_50Hz_550nW_15.2W_004.ptu"
    assert model.sort_value(e, "power_density_w_cm2") == 15.2
    js = next(e for e in entries if e.sample == "JS249")
    assert model.display_value(js, "precipitation") == ""
    assert model.sort_value(js, "kpfm_amplitude_v") in (1.0, float("inf"))
    assert model.sort_value(js, "total_power_nw") == float("inf")   # None sorts last


def test_text_search_matches_all_words_case_insensitively(entries):
    assert {e.dr for e in model.apply_filters(entries, "cf501")} == {"DR2", "DR3"}
    assert len(model.apply_filters(entries, "CF501 DR3")) == 1
    assert len(model.apply_filters(entries, "8x precipitated")) == 2
    assert model.apply_filters(entries, "CF501 532") == []
    assert len(model.apply_filters(entries, "   ")) == len(entries)


def test_distinct_values_and_quick_filters(entries):
    options = model.distinct_values(entries, "wavelength_nm")
    assert [label for _, label in options] == ["440", "532"]
    precip = model.distinct_values(entries, "precipitation")
    assert precip[-1] == (model.NONE_VALUE, "(none)")

    assert {e.sample for e in model.apply_filters(
        entries, criteria={"precipitation": model.NONE_VALUE})} == {"JS249"}
    selected = model.apply_filters(entries, criteria={"wavelength_nm": 532.0, "sample": None})
    assert len(selected) == 1 and selected[0].sample == "JS249"
    assert len(model.apply_filters(entries, "DR1", {"sample": "JS249"})) == 2


def test_missing_files(entries, archive):
    e = entries[0]
    assert model.missing_files(e) == []
    e.ptu_path.unlink()
    assert model.missing_files(e) == [e.ptu_path]


def test_document_tracks_unsaved_changes(archive, tmp_path):
    doc = model.DatabaseDocument()
    doc.build(archive)
    assert doc.dirty and doc.root == archive and len(doc.entries) == 4

    path = tmp_path / "db.json"
    doc.save(path)
    assert not doc.dirty and doc.path == path
    assert len(load_json(path)) == 4

    assert doc.remove([doc.entries[0]]) == 1
    assert doc.dirty and len(doc.entries) == 3

    doc.open(path)
    assert not doc.dirty and len(doc.entries) == 4 and doc.root is None


def test_document_add_uses_archive_root_when_inside(archive, tmp_path):
    doc = model.DatabaseDocument()
    doc.build(archive)
    base = archive / "CF501" / "4x precipitated" / "06.02.2026"
    _touch(base / "06.02_DR1_25Hz_hKPFM2V.txt")
    _touch(base / "DR1_360s_25Hz_10W_002.ptu")
    entry = doc.add(base / "06.02_DR1_25Hz_hKPFM2V.txt", base / "DR1_360s_25Hz_10W_002.ptu")
    assert (entry.sample, entry.precipitation, entry.date) == (
        "CF501", "4x precipitated", "06.02.2026")

    outside = tmp_path / "elsewhere"
    _touch(outside / "DR5_25Hz_hKPFM2V.txt")
    _touch(outside / "DR5_360s_25Hz_10W_000.ptu")
    entry = doc.add(outside / "DR5_25Hz_hKPFM2V.txt", outside / "DR5_360s_25Hz_10W_000.ptu")
    assert entry.sample == "elsewhere"
    assert len(doc.entries) == 6


# -------------------------------------------------------------------- tab
qt = pytest.importorskip("pytestqt")


@pytest.fixture
def tab(qtbot, tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QMessageBox

    from kpfm_pl.gui.context import AppContext
    from kpfm_pl.gui.tabs.database_tab import DatabaseTab

    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    ctx = AppContext(settings=settings)
    widget = DatabaseTab(ctx)
    qtbot.addWidget(widget)
    # Answer every confirmation with "yes/discard" unless a test overrides it.
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Discard
                                     if len(a) > 3 else QMessageBox.StandardButton.Yes))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    yield widget
    ctx.runner.cancel_all()
    ctx.runner.wait(10000)


def _build(tab, qtbot, archive, monkeypatch):
    monkeypatch.setattr(tab.ctx, "open_folder_dialog", lambda *a, **k: str(archive))
    tab.build_from_folder()
    qtbot.waitUntil(lambda: not tab.ctx.runner.busy and bool(tab.doc.entries), timeout=10000)


def test_tab_builds_filters_and_saves(tab, qtbot, archive, tmp_path, monkeypatch):
    _build(tab, qtbot, archive, monkeypatch)
    assert tab.table_model.rowCount() == 4
    assert tab.proxy.rowCount() == 4
    assert "unsaved" in tab.status_label.text()

    tab.search.setText("CF501")
    assert tab.proxy.rowCount() == 2
    assert tab.count_label.text() == "2 of 4 entries"
    tab.reset_filters()
    assert tab.proxy.rowCount() == 4

    combo = tab.filter_form.fields["wavelength_nm"]
    combo.setCurrentIndex(combo.findData(532.0))
    assert [e.sample for e in tab.visible_entries()] == ["JS249"]

    csv_path = tmp_path / "table.csv"
    monkeypatch.setattr(tab.ctx, "save_file_dialog", lambda *a, **k: str(csv_path))
    tab.export_csv()
    assert len(csv_path.read_text(encoding="utf-8").strip().splitlines()) == 2  # header + 1

    json_path = tmp_path / "db"
    monkeypatch.setattr(tab.ctx, "save_file_dialog", lambda *a, **k: str(json_path))
    tab.save_database()
    assert (tmp_path / "db.json").is_file()
    assert not tab.doc.dirty
    assert tab.reopen_button.isEnabled()


def test_tab_sorts_numeric_columns_numerically(tab, qtbot, archive, monkeypatch):
    from PySide6.QtCore import Qt

    _build(tab, qtbot, archive, monkeypatch)
    column = next(i for i, c in enumerate(model.COLUMNS) if c.field == "pid_bandwidth_hz")
    tab.table.sortByColumn(column, Qt.SortOrder.AscendingOrder)
    values = [e.pid_bandwidth_hz for e in tab.visible_entries()]
    assert values == sorted(values)


def test_tab_remove_and_reopen(tab, qtbot, archive, tmp_path, monkeypatch):
    _build(tab, qtbot, archive, monkeypatch)
    path = tmp_path / "db.json"
    monkeypatch.setattr(tab.ctx, "save_file_dialog", lambda *a, **k: str(path))
    tab.save_database()

    tab.table.selectRow(0)
    tab.remove_selected()
    assert len(tab.doc.entries) == 3 and tab.doc.dirty

    tab.reopen_last()
    assert len(tab.doc.entries) == 4 and not tab.doc.dirty


def test_tab_keeps_unsaved_changes_when_user_cancels(tab, qtbot, archive, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    _build(tab, qtbot, archive, monkeypatch)
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel))
    asked = []
    monkeypatch.setattr(tab.ctx, "open_file_dialog", lambda *a, **k: asked.append(1))
    tab.open_database()
    assert asked == [] and len(tab.doc.entries) == 4


def test_tab_open_selected_loads_pair_and_switches_tab(tab, qtbot, archive, monkeypatch):
    _build(tab, qtbot, archive, monkeypatch)
    loads, shown = [], []
    monkeypatch.setattr(tab.ctx, "load_file", lambda kind, path, **k: loads.append((kind, path)))
    monkeypatch.setattr(tab.ctx, "show_tab", shown.append)
    tab.table.selectRow(0)
    entry = tab.selected_entries()[0]
    tab.open_selected()
    assert loads == [("zurich", entry.txt_path), ("ptu", entry.ptu_path)]
    assert shown == ["CorrelateTab"]

    loads.clear()
    entry.txt_path.unlink()
    tab.open_selected()
    assert loads == []  # missing file -> warning instead of loading


def test_tab_add_pair(tab, qtbot, archive, monkeypatch):
    _build(tab, qtbot, archive, monkeypatch)
    base = archive / "JS249" / "02.03.2026"
    _touch(base / "JS249_02,03_DR2_1Hz_AM17_Scan1.txt")
    _touch(base / "DR2_360s_1Hz_4.67W_Scan1_000.ptu")
    picks = iter([str(base / "JS249_02,03_DR2_1Hz_AM17_Scan1.txt"),
                  str(base / "DR2_360s_1Hz_4.67W_Scan1_000.ptu")])
    monkeypatch.setattr(tab.ctx, "open_file_dialog", lambda *a, **k: next(picks))
    tab.add_pair()
    assert tab.table_model.rowCount() == 5
    added = tab.doc.entries[-1]
    assert (added.sample, added.date, added.dr) == ("JS249", "02.03.2026", "DR2")
