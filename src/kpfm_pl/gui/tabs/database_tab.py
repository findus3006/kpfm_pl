"""Database tab: build, filter and edit the measurement database; open pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QTableView,
)

from ...database import Entry, build_database
from ..forms import ParamForm
from ..models import database as model
from .base import BaseTab

__all__ = ["DatabaseTab", "EntryTableModel"]

#: Item-data role returning the :class:`Entry` of a row.
ENTRY_ROLE = Qt.ItemDataRole.UserRole + 1
#: Item-data role returning the sort key of a cell.
SORT_ROLE = Qt.ItemDataRole.UserRole

_SETTINGS_KEY = "database/last_json"


class EntryTableModel(QAbstractTableModel):
    """Table of database entries, one row per measurement pair."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.entries: list[Entry] = []

    def set_entries(self, entries: list[Entry]) -> None:
        self.beginResetModel()
        self.entries = list(entries)
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.entries)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(model.COLUMNS)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        entry = self.entries[index.row()]
        column = model.COLUMNS[index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return model.display_value(entry, column.field)
        if role == SORT_ROLE:
            return model.sort_value(entry, column.field)
        if role == ENTRY_ROLE:
            return entry
        if role == Qt.ItemDataRole.ToolTipRole and column.field in ("txt_path", "ptu_path"):
            return str(getattr(entry, column.field))
        if role == Qt.ItemDataRole.TextAlignmentRole and column.numeric:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section: int, orientation: Qt.Orientation,
                   role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if orientation != Qt.Orientation.Horizontal:
            return None
        column = model.COLUMNS[section]
        if role == Qt.ItemDataRole.DisplayRole:
            return column.header
        if role == Qt.ItemDataRole.ToolTipRole:
            return column.tooltip
        return None


class _FilterProxy(QSortFilterProxyModel):
    """Shows only the rows whose entries are in :attr:`accepted`."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.accepted: set[int] | None = None   # ids of visible entries; None = all
        self.setSortRole(SORT_ROLE)

    def set_accepted(self, entries: list[Entry] | None) -> None:
        # Qt >= 6.9 replaces invalidateFilter() by begin/endFilterChange().
        if hasattr(self, "beginFilterChange"):
            self.beginFilterChange()
        self.accepted = None if entries is None else {id(e) for e in entries}
        if hasattr(self, "endFilterChange"):
            self.endFilterChange(QSortFilterProxyModel.Direction.Rows)
        else:
            self.invalidateFilter()

    def filterAcceptsRow(self, row: int, parent: QModelIndex) -> bool:
        if self.accepted is None:
            return True
        entry = self.sourceModel().entries[row]
        return id(entry) in self.accepted


class DatabaseTab(BaseTab):
    """Index a measurement archive and open KPFM/PL pairs from it."""

    title = "Database"

    def __init__(self, ctx, parent=None) -> None:
        super().__init__(ctx, parent)
        self.doc = model.DatabaseDocument()

        self.add_hint(
            "1. Build the database from your measurement folder "
            "(<sample>/[<precipitation>/]<date>/) or open a saved one.\n"
            "2. Narrow the table with the search box and the filters.\n"
            "3. Double-click a row (or press 'Open selected pair') to load its "
            ".txt and .ptu files into the Correlate tab.\n"
            "Save the database as JSON to reopen it later without rescanning."
        )
        self._build_file_group()
        self._build_filter_group()
        self._build_edit_group()

        self.table_model = EntryTableModel(self)
        self.proxy = _FilterProxy(self)
        self.proxy.setSourceModel(self.table_model)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(-1, Qt.SortOrder.AscendingOrder)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.doubleClicked.connect(lambda _index: self.open_selected())
        self.table.selectionModel().selectionChanged.connect(
            lambda *_: self._update_buttons())
        self.content_layout.addWidget(self.table)

        self._refresh()

    # ------------------------------------------------------------- controls
    def _build_file_group(self) -> None:
        layout = self.add_group("Database")
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.build_button = self.add_button(
            layout, "Build from folder…", self.build_from_folder, primary=True,
            tooltip="Scan a measurement folder for .txt/.ptu pairs. Pairs that cannot "
                    "be matched are reported in the log.")
        self.add_button(layout, "Open database (JSON)…", self.open_database,
                        tooltip="Open a database saved earlier.")
        self.reopen_button = self.add_button(
            layout, "Reopen last database", self.reopen_last,
            tooltip="Open the database used most recently.")
        self.save_button = self.add_button(
            layout, "Save database (JSON)…", self.save_database,
            tooltip="Save all entries as JSON, to reopen them later without rescanning.")
        self.csv_button = self.add_button(
            layout, "Export table (CSV)…", self.export_csv,
            tooltip="Save the entries currently shown in the table as CSV (e.g. for Excel).")

    def _build_filter_group(self) -> None:
        layout = self.add_group("Filter")
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search, e.g. CF501 DR2 532")
        self.search.setClearButtonEnabled(True)
        self.search.setToolTip("Shows entries containing every word, in any column "
                               "or in the file paths.")
        self.search.textChanged.connect(lambda _: self._apply_filters())
        layout.addWidget(self.search)
        f = self.filter_form = ParamForm()
        for field in model.QUICK_FILTERS:
            column = next(c for c in model.COLUMNS if c.field == field)
            f.add_choice(field, column.header, [(None, "All")],
                         tooltip=f"Only show entries with this {column.header.lower()}.")
        f.changed.connect(self._apply_filters)
        layout.addWidget(f)
        self.count_label = QLabel()
        layout.addWidget(self.count_label)
        self.add_button(layout, "Reset filters", self.reset_filters)

    def _build_edit_group(self) -> None:
        layout = self.add_group("Entries")
        self.open_button = self.add_button(
            layout, "Open selected pair", self.open_selected, primary=True,
            tooltip="Load the selected entry's .txt and .ptu files and switch to the "
                    "Correlate tab. Double-clicking a row does the same.")
        self.add_button(layout, "Add pair…", self.add_pair,
                        tooltip="Add a .txt/.ptu pair that is not in the database yet.")
        self.remove_button = self.add_button(
            layout, "Remove selected", self.remove_selected,
            tooltip="Remove the selected entries from the database (files are not deleted).")

    # ------------------------------------------------------------- state
    def _last_path(self) -> Path | None:
        value = self.ctx.settings.value(_SETTINGS_KEY, "")
        return Path(value) if value else None

    def _refresh(self) -> None:
        """Push the document into the table and update labels and filters."""
        self.table_model.set_entries(self.doc.entries)
        for field in model.QUICK_FILTERS:
            self.filter_form.set_choices(
                field, [(None, "All"), *model.distinct_values(self.doc.entries, field)])
        self._apply_filters()
        self.table.resizeColumnsToContents()
        self._update_status()

    def _update_status(self) -> None:
        if self.doc.path is not None:
            source = self.doc.path.name
        elif self.doc.root is not None:
            source = f"built from {self.doc.root.name}"
        else:
            source = "no database open"
        unsaved = " (unsaved changes)" if self.doc.dirty else ""
        self.status_label.setText(f"{len(self.doc.entries)} entries, {source}{unsaved}")
        tip = str(self.doc.path or self.doc.root or "")
        self.status_label.setToolTip(tip)
        last = self._last_path()
        self.reopen_button.setEnabled(last is not None and last.is_file())
        self.reopen_button.setToolTip(f"Open {last}" if last else "No database used yet.")
        self._update_buttons()

    def _update_buttons(self) -> None:
        has_entries = not self.doc.is_empty
        self.save_button.setEnabled(has_entries)
        self.csv_button.setEnabled(has_entries)
        selected = bool(self.selected_entries())
        self.open_button.setEnabled(selected)
        self.remove_button.setEnabled(selected)

    def _apply_filters(self) -> None:
        criteria = self.filter_form.values()
        visible = model.apply_filters(self.doc.entries, self.search.text(), criteria)
        self.proxy.set_accepted(visible)
        self.count_label.setText(f"{len(visible)} of {len(self.doc.entries)} entries")
        self._update_buttons()

    def visible_entries(self) -> list[Entry]:
        """Entries shown in the table, in display order."""
        return [self.proxy.index(row, 0).data(ENTRY_ROLE)
                for row in range(self.proxy.rowCount())]

    def selected_entries(self) -> list[Entry]:
        rows = self.table.selectionModel().selectedRows() if self.table.model() else []
        return [index.data(ENTRY_ROLE) for index in sorted(rows, key=lambda i: i.row())]

    def reset_filters(self) -> None:
        self.search.clear()
        for field in model.QUICK_FILTERS:
            self.filter_form.fields[field].setCurrentIndex(0)   # "All"

    def _confirm_discard(self) -> bool:
        """Ask before replacing a database with unsaved changes."""
        if not self.doc.dirty:
            return True
        answer = QMessageBox.question(
            self, "Unsaved changes",
            "The current database has unsaved changes. Discard them?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        return answer == QMessageBox.StandardButton.Discard

    # -------------------------------------------------------------- actions
    def build_from_folder(self) -> None:
        if not self._confirm_discard():
            return
        root = self.ctx.open_folder_dialog("Choose the measurement folder")
        if not root:
            return

        def _built(entries) -> None:
            self.doc.set_built(entries, root)
            self._refresh()
            self.ctx.log.info(f"Database built from {root}: {len(entries)} pairs. "
                              "Save it to reopen it later.")

        self.run_job(f"Building database from {Path(root).name}",
                     lambda progress: build_database(root), _built,
                     disable=[self.build_button])

    def open_database(self) -> None:
        if not self._confirm_discard():
            return
        path = self.ctx.open_file_dialog("Open database", "Database (*.json)")
        if path:
            self._open(path)

    def reopen_last(self) -> None:
        last = self._last_path()
        if last is None or not self._confirm_discard():
            return
        self._open(last)

    def _open(self, path: str | Path) -> None:
        try:
            self.doc.open(path)
        except Exception as exc:
            QMessageBox.warning(self, "Open database",
                                f"Could not open {Path(path).name}:\n{exc}")
            return
        self.ctx.settings.setValue(_SETTINGS_KEY, str(path))
        self.ctx.log.info(f"Opened database {path} ({len(self.doc.entries)} entries)")
        self._refresh()

    def save_database(self) -> None:
        default = self.doc.path.name if self.doc.path else "database.json"
        path = self.ctx.save_file_dialog("Save database", default, "Database (*.json)")
        if not path:
            return
        path = Path(path)
        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
        try:
            self.doc.save(path)
        except Exception as exc:
            QMessageBox.warning(self, "Save database", f"Could not save the database:\n{exc}")
            return
        self.ctx.settings.setValue(_SETTINGS_KEY, str(path))
        self.ctx.log.info(f"Database saved: {path}")
        self._update_status()

    def export_csv(self) -> None:
        entries = self.visible_entries()
        if not entries:
            QMessageBox.information(self, "Export table", "The table is empty.")
            return
        path = self.ctx.save_file_dialog("Export table", "database.csv", "CSV file (*.csv)")
        if not path:
            return
        path = Path(path)
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        try:
            self.doc.export_csv(entries, path)
        except Exception as exc:
            QMessageBox.warning(self, "Export table", f"Could not export the table:\n{exc}")
            return
        self.ctx.log.info(f"Table exported ({len(entries)} entries): {path}")

    def add_pair(self) -> None:
        txt = self.ctx.open_file_dialog("Choose the KPFM file (.txt)",
                                        "Zurich lock-in export (*.txt)")
        if not txt:
            return
        ptu = self.ctx.open_file_dialog("Choose the matching PL file (.ptu)",
                                        "PicoQuant PTU (*.ptu)")
        if not ptu:
            return
        try:
            entry = self.doc.add(txt, ptu)
        except Exception as exc:
            QMessageBox.warning(self, "Add pair", f"Could not add the pair:\n{exc}")
            return
        self.ctx.log.info(f"Added {Path(txt).name} + {Path(ptu).name} "
                          f"(sample {entry.sample})")
        self._refresh()

    def remove_selected(self) -> None:
        entries = self.selected_entries()
        if not entries:
            return
        answer = QMessageBox.question(
            self, "Remove entries",
            f"Remove {len(entries)} selected entr{'y' if len(entries) == 1 else 'ies'} "
            "from the database? The files themselves are not deleted.")
        if answer != QMessageBox.StandardButton.Yes:
            return
        removed = self.doc.remove(entries)
        self.ctx.log.info(f"Removed {removed} entries from the database.")
        self._refresh()

    def open_selected(self) -> None:
        entries = self.selected_entries()
        if not entries:
            return
        entry = entries[0]
        missing = model.missing_files(entry)
        if missing:
            QMessageBox.warning(
                self, "Open pair",
                "These files of the selected entry were not found:\n\n"
                + "\n".join(str(p) for p in missing)
                + "\n\nThey may have been moved or renamed. Rebuild the database "
                  "from the folder to update it.")
            return
        self.ctx.load_file("zurich", entry.txt_path)
        self.ctx.load_file("ptu", entry.ptu_path)
        self.ctx.show_tab("CorrelateTab")
