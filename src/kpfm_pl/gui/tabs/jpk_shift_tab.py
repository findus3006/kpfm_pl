"""AFM shift tab: the same circle on up to four JPK scans, plus the measured shift."""

from __future__ import annotations

from matplotlib.patches import Circle
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QTableWidget,
    QTableWidgetItem,
)

from ...plotting import plot_image
from ...selector import DraggableCircle
from ..canvas import PlotPanel
from ..forms import ParamForm
from ..models import jpk as jpk_model
from ..models import jpk_shift as model
from ..models.files import KINDS, load_jpk
from .base import BaseTab

__all__ = ["JPKShiftTab"]

_SHIFT_COLUMNS = ("file", "source", "dx", "dy", "distance", "peak")
_STAT_COLUMNS = ("file", "channel", "unit", "pixels", "mean", "std", "min", "max")
_REFERENCE_COLOUR = "deepskyblue"


def _fill_table(table: QTableWidget, rows, columns) -> None:
    table.setRowCount(0)
    for row in rows:
        index = table.rowCount()
        table.insertRow(index)
        for col, key in enumerate(columns):
            value = row.get(key)
            if value is None:
                text = ""
            elif isinstance(value, float):
                text = f"{value:.4g}"
            else:
                text = str(value)
            table.setItem(index, col, QTableWidgetItem(text))
    table.resizeColumnsToContents()


def _checkable(name: str, checked: bool) -> QListWidgetItem:
    item = QListWidgetItem(name)
    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
    item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
    return item


def _checked(widget: QListWidget) -> list[str]:
    items = (widget.item(i) for i in range(widget.count()))
    return [item.text() for item in items if item.checkState() == Qt.CheckState.Checked]


class JPKShiftTab(BaseTab):
    """Compare up to four ``.jpk`` scans with one circle to see how much they shifted."""

    title = "AFM shift"

    def __init__(self, ctx, parent=None) -> None:
        super().__init__(ctx, parent)
        self.handles: list = []                          # LoadedFile per row
        self.ref = 0                                     # index of the reference file
        self.images: list[dict] = []                     # per file: channel -> Image
        self.styles: dict[str, jpk_model.ChannelStyle] = {}
        self.process = True
        self.circles: list[jpk_model.CircleROI | None] = []   # per file, in metres
        self.auto_results: list[model.ShiftResult] = []
        self.results: list[model.ShiftResult] = []
        self.stats: list[dict] = []
        self._auto_settings: tuple | None = None         # (region, ref circle) of auto_results
        self._axes: dict = {}                            # (file index, channel) -> Axes
        self._selectors: dict[int, DraggableCircle] = {}
        self._overlays: list = []
        self._cids: list[int] = []

        self.add_hint(
            f"1. Add up to {model.MAX_FILES} .jpk scans and pick the reference. Each "
            "row of the plot is one file, each column one channel.\n"
            "2. Draw a circle on the reference (left-drag on the 'Draw on' channel). It "
            "is copied to the same scan position on every other file; the blue dotted "
            "circle marks that position.\n"
            "3. Drag each file's circle onto the feature: the table gives the shift. Or "
            "click 'Detect shift' to cross-correlate the scans; 'Move circles to "
            "detected shift' places the circles accordingly.\n"
            "Right-click a circle: on the reference it is deleted, on another file it "
            "jumps back to the reference position.\nHover over a field for an "
            "explanation."
        )
        self._build_file_group()
        self._build_channel_group()
        self._build_display_group()
        self._build_circle_group()
        self._build_shift_group()
        self._build_stats_group()

        self.plot = PlotPanel(ctx, navigation="minimal", figsize=(10, 8))
        self.plot.set_export_provider(self._export_payloads)
        self.plot.set_figure_provider(self._figure_info)
        self.plot.show_message("Add .jpk files to start.")
        self.content_layout.addWidget(self.plot)
        self._update_enabled()

    # ------------------------------------------------------------- controls
    def _build_file_group(self) -> None:
        layout = self.add_group("Files")
        self.file_list = QListWidget()
        self.file_list.setToolTip(f"Loaded scans (at most {model.MAX_FILES}). They are "
                                  "plotted in this order, one row each.")
        self.file_list.setMinimumHeight(90)
        layout.addWidget(self.file_list)
        f = self.file_form = ParamForm()
        f.add_choice("reference", "Reference", [(0, "")],
                     tooltip="The circle is drawn on this file and every shift is "
                             "measured relative to it.")
        f.fields["reference"].currentIndexChanged.connect(lambda _: self._reference_changed())
        layout.addWidget(f)
        row = QHBoxLayout()
        layout.addLayout(row)
        self.add_files_button = self.add_button(
            row, "Add files…", self.add_files_dialog,
            tooltip="Open one or more .jpk files.")
        self.add_current_button = self.add_button(
            row, "Add from AFM tab", self.add_current_file,
            tooltip="Add the file that is open in the 'AFM (JPK)' tab.")
        self.remove_file_button = self.add_button(
            layout, "Remove selected file", self.remove_selected_file,
            tooltip="Remove the file selected in the list above.")

    def _build_channel_group(self) -> None:
        layout = self.add_group("Channels")
        self.channel_list = QListWidget()
        self.channel_list.setToolTip("Channels present in every file. Tick the ones to "
                                     "plot; click a name to edit its display below.")
        self.channel_list.setMinimumHeight(140)
        self.channel_list.itemChanged.connect(lambda _: self.reload())
        self.channel_list.currentItemChanged.connect(lambda *_: self._show_style())
        layout.addWidget(self.channel_list)
        f = self.channel_form = ParamForm()
        f.add_bool("process", "Processing", True, text="line levelling",
                   tooltip="Line-level and zero height channels and convert "
                           "Precision 6 from volts to newtons.\nUnchecked: raw "
                           "calibrated data.")
        f.add_choice("extent_unit", "Axes unit", jpk_model.EXTENT_UNITS, "um",
                     tooltip="Length unit of the image axes, the circle and the shifts.")
        f.add_bool("shared_range", "Colour range", True, text="same for all files",
                   tooltip="Use one colour range per channel for all files, so the "
                           "maps can be compared directly.")
        f.fields["process"].toggled.connect(lambda _: self.reload())
        f.fields["extent_unit"].currentIndexChanged.connect(lambda _: self._unit_changed())
        f.fields["shared_range"].toggled.connect(lambda _: self._draw())
        layout.addWidget(f)

    def _build_display_group(self) -> None:
        layout = self.add_group("Display of selected channel", collapsible=True,
                                collapsed=True)
        self.style_label = QLabel("Click a channel name above.")
        self.style_label.setWordWrap(True)
        layout.addWidget(self.style_label)
        f = self.style_form = ParamForm()
        f.add_choice("cmap", "Colormap", jpk_model.COLORMAPS, "viridis",
                     tooltip="Colour scale of this channel (all files).")
        f.add_choice("unit", "Colorbar unit", [("", "")],
                     tooltip="Unit of the colorbar and of the statistics table.")
        f.fields["cmap"].currentIndexChanged.connect(lambda _: self._style_changed())
        f.fields["unit"].currentIndexChanged.connect(lambda _: self._style_changed())
        f.setEnabled(False)
        layout.addWidget(f)

    def _build_circle_group(self) -> None:
        layout = self.add_group("Circle")
        f = self.circle_form = ParamForm()
        f.add_choice("draw_on", "Draw on", [("", "")],
                     tooltip="Channel on which the circles are drawn and moved with the "
                             "mouse (in every file).")
        f.add_float("x", "Centre x", 0.0, minimum=-1e6, maximum=1e6, decimals=4,
                    tooltip="Centre of the reference circle in the axes unit.")
        f.add_float("y", "Centre y", 0.0, minimum=-1e6, maximum=1e6, decimals=4,
                    tooltip="Centre of the reference circle in the axes unit.")
        f.add_float("radius", "Radius", 0.1, minimum=0.0, maximum=1e6, decimals=4,
                    tooltip="Radius of all circles in the axes unit.")
        f.add_bool("show_reference", "Reference position", True, text="mark on other files",
                   tooltip="Draw the reference circle as a blue dotted circle on the "
                           "other files, so the shift is visible at a glance.")
        f.fields["draw_on"].currentIndexChanged.connect(lambda _: self._build_overlays())
        f.fields["show_reference"].toggled.connect(lambda _: self._build_overlays())
        layout.addWidget(f)
        layout.addWidget(QLabel("Show the circle also on:"))
        self.show_on_list = QListWidget()
        self.show_on_list.setToolTip("Channels on which the circle of each file is copied "
                                     "(dashed). The 'Draw on' channel always shows it.")
        self.show_on_list.setMaximumHeight(100)
        self.show_on_list.itemChanged.connect(lambda _: self._build_overlays())
        layout.addWidget(self.show_on_list)
        row = QHBoxLayout()
        layout.addLayout(row)
        self.set_circle_button = self.add_button(
            row, "Set circle", self.apply_circle_fields,
            tooltip="Place the reference circle at the centre and radius entered above.")
        self.remove_circle_button = self.add_button(
            row, "Remove circle", self.remove_circle, tooltip="Delete all circles.")
        self.reset_circles_button = self.add_button(
            layout, "Reset file circles to reference", self.reset_circles,
            tooltip="Move the circles of all other files back onto the reference circle.")

    def _build_shift_group(self) -> None:
        layout = self.add_group("Shift")
        f = self.shift_form = ParamForm()
        f.add_choice("region", "Detect on", model.REGIONS, "image",
                     tooltip="Whole image: overall drift between the scans.\nAround the "
                             "circle: drift of the feature in the reference circle, from "
                             "a square window around it.")
        f.add_float("window", "Window", 3.0, minimum=1.0, maximum=50.0, decimals=1,
                    suffix=" × diameter",
                    tooltip="Side of the square window around the circle, in circle "
                            "diameters. Shifts larger than about half the window cannot "
                            "be detected.")
        f.add_choice("use_channel", "Use shift of", [("", "")],
                     tooltip="Channel whose detected shift 'Move circles to detected "
                             "shift' applies.")
        f.fields["region"].currentIndexChanged.connect(lambda _: self._region_changed())
        layout.addWidget(f)
        self.detect_button = self.add_button(
            layout, "Detect shift", self.detect_shift, primary=True,
            tooltip="Cross-correlate every plotted channel of each file with the "
                    "reference. 'correlation' close to 1 = reliable, close to 0 = not.")
        self.apply_button = self.add_button(
            layout, "Move circles to detected shift", self.apply_detected,
            tooltip="Place each file's circle at the reference circle plus the detected "
                    "shift of the channel chosen in 'Use shift of'.")
        self.shift_table = QTableWidget(0, len(_SHIFT_COLUMNS))
        self.shift_table.setHorizontalHeaderLabels(["file", "source", "dx", "dy", "|d|",
                                                    "correlation"])
        self.shift_table.verticalHeader().setVisible(False)
        self.shift_table.setMinimumHeight(130)
        self.shift_table.setToolTip("Shift of each file relative to the reference in the "
                                    "axes unit (x to the right, y up).\nsource 'circle' = "
                                    "your circle; a channel name = automatic detection.")
        layout.addWidget(self.shift_table)
        self.copy_shift_button = self.add_button(
            layout, "Copy shift table", self.copy_shifts,
            tooltip="Copy the table (tab-separated) to paste into Excel or Origin.")
        self._region_changed()

    def _build_stats_group(self) -> None:
        layout = self.add_group("Circle statistics", collapsible=True)
        self.stats_table = QTableWidget(0, len(_STAT_COLUMNS))
        self.stats_table.setHorizontalHeaderLabels(list(_STAT_COLUMNS))
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setMinimumHeight(130)
        self.stats_table.setToolTip("Statistics inside each file's own circle, in each "
                                    "channel's colorbar unit.")
        layout.addWidget(self.stats_table)
        self.copy_stats_button = self.add_button(
            layout, "Copy statistics", self.copy_stats,
            tooltip="Copy the table (tab-separated) to paste into Excel or Origin.")

    def _update_enabled(self) -> None:
        has_files = bool(self.handles)
        has_images = bool(self.images)
        for widget in (self.channel_list, self.channel_form, self.file_form,
                       self.remove_file_button):
            widget.setEnabled(has_files)
        for widget in (self.circle_form, self.show_on_list, self.set_circle_button,
                       self.remove_circle_button, self.reset_circles_button,
                       self.copy_stats_button):
            widget.setEnabled(has_images)
        self.detect_button.setEnabled(has_images and len(self.handles) > 1)
        self.apply_button.setEnabled(bool(self.auto_results))
        self.copy_shift_button.setEnabled(bool(self.results))
        full = len(self.handles) >= model.MAX_FILES
        self.add_files_button.setEnabled(not full)
        self.add_current_button.setEnabled(not full)

    # ---------------------------------------------------------------- state
    @property
    def extent_unit(self) -> str:
        return self.channel_form.value("extent_unit")

    @property
    def labels(self) -> list[str]:
        return model.file_labels(self.handles)

    def checked_channels(self) -> list[str]:
        return _checked(self.channel_list)

    def set_checked_channels(self, channels) -> None:
        """Tick exactly ``channels`` (and reload once)."""
        self.channel_list.blockSignals(True)
        for i in range(self.channel_list.count()):
            item = self.channel_list.item(i)
            item.setCheckState(Qt.CheckState.Checked if item.text() in channels
                               else Qt.CheckState.Unchecked)
        self.channel_list.blockSignals(False)
        self.reload()

    def _plotted_channels(self) -> list[str]:
        return list(self.images[0]) if self.images else []

    # ---------------------------------------------------------------- files
    def add_files_dialog(self) -> None:
        paths = self.ctx.open_files_dialog("Open JPK files", KINDS["jpk"])
        if paths:
            self.add_files(paths)

    def add_files(self, paths) -> None:
        """Load ``paths`` in the background and append them (up to the file limit)."""
        paths = list(paths)
        room = model.MAX_FILES - len(self.handles)
        if len(paths) > room:
            self.ctx.log.warning(f"At most {model.MAX_FILES} files can be compared; "
                                 f"{len(paths) - room} file(s) were not added.")
            paths = paths[:max(room, 0)]
        if not paths:
            return

        def _load(progress):
            handles = []
            for i, path in enumerate(paths):
                handles.append(load_jpk(path))
                progress(i + 1, len(paths))
            return handles

        self.run_job(f"Loading {len(paths)} JPK file(s)", _load, self._add_handles,
                     disable=[self.add_files_button, self.add_current_button],
                     cancellable=True)

    def add_current_file(self) -> None:
        handle = self.ctx.state.jpk
        if handle is None:
            self.ctx.log.warning("No file is open in the 'AFM (JPK)' tab.")
            return
        self._add_handles([handle])

    def _add_handles(self, handles) -> None:
        room = model.MAX_FILES - len(self.handles)
        handles = list(handles)[:max(room, 0)]
        if not handles:
            return
        reference = self.circles[self.ref] if self.circles else None
        for handle in handles:
            self.handles.append(handle)
            self.circles.append(None if reference is None else
                                jpk_model.CircleROI(reference.x, reference.y,
                                                    reference.radius))
            self.ctx.log.info(f"AFM shift: added {handle.name}")
        self._files_changed()

    def remove_selected_file(self) -> None:
        row = self.file_list.currentRow()
        if not 0 <= row < len(self.handles):
            return
        self.handles.pop(row)
        self.circles.pop(row)
        if row == self.ref:
            self.ref = 0
            # The new reference keeps its circle; files that still sat on the old
            # reference position stay where they are.
        elif row < self.ref:
            self.ref -= 1
        self._files_changed()

    def _files_changed(self) -> None:
        labels = self.labels
        self.file_list.clear()
        for i, label in enumerate(labels):
            self.file_list.addItem(label + ("   [reference]" if i == self.ref else ""))
        self.file_form.blockSignals(True)
        combo = self.file_form.fields["reference"]
        combo.blockSignals(True)
        self.file_form.set_choices("reference", [(i, label) for i, label in enumerate(labels)]
                                   or [(0, "")])
        combo.setCurrentIndex(self.ref if labels else 0)
        combo.blockSignals(False)
        self.file_form.blockSignals(False)
        self._clear_auto()

        # Channels present in every file; keep the current ticks where possible.
        previous = self.checked_channels()
        names = model.common_channels(self.handles)
        wanted = [n for n in previous if n in names] or jpk_model.default_channels(names)
        self.channel_list.blockSignals(True)
        self.channel_list.clear()
        for name in names:
            self.channel_list.addItem(_checkable(name, name in wanted))
            if name not in self.styles:
                unit = self.handles[self.ref].reader.units.get(name, "")
                self.styles[name] = jpk_model.default_style(name, unit)
        self.channel_list.blockSignals(False)
        if names and self.channel_list.currentRow() < 0:
            self.channel_list.setCurrentRow(names.index(wanted[0]) if wanted else 0)
        self._update_enabled()
        self.reload()

    def _reference_changed(self) -> None:
        index = self.file_form.fields["reference"].currentIndex()
        if not 0 <= index < len(self.handles) or index == self.ref:
            return
        old = self.circles[self.ref]
        self.ref = index
        if self.circles[index] is None and old is not None:
            self.circles[index] = jpk_model.CircleROI(old.x, old.y, old.radius)
        for i in range(self.file_list.count()):
            self.file_list.item(i).setText(
                self.labels[i] + ("   [reference]" if i == self.ref else ""))
        self._clear_auto()
        self._draw()

    # --------------------------------------------------------------- loading
    def reload(self) -> None:
        handles = list(self.handles)
        channels = self.checked_channels()
        process = self.channel_form.value("process")
        if not handles or not channels:
            self.images = []
            self._clear_auto()
            self._refresh_choices()
            self._draw()
            return

        def _done(value) -> None:
            if handles != self.handles or channels != self.checked_channels():
                return          # the selection changed meanwhile; a newer load follows
            self._images_loaded(value)

        self.run_job(f"Reading {len(channels)} channel(s) of {len(handles)} file(s)",
                     lambda progress: (model.load_all(handles, channels, process, progress),
                                       process),
                     _done, disable=[self.channel_list], cancellable=True)

    def _images_loaded(self, value) -> None:
        images, self.process = value
        # Precision 6 is returned in newtons when processed: adapt the default style.
        if images:
            for name, image in images[self.ref].items():
                if image.units["z"] and self.styles[name].unit not in dict(
                        jpk_model.unit_options(image.units["z"])):
                    self.styles[name] = jpk_model.default_style(name, image.units["z"])
        self.images = images
        self._clear_auto()
        self._refresh_choices()
        self._show_style()
        self._draw()

    def _refresh_choices(self) -> None:
        channels = self._plotted_channels()
        self.circle_form.blockSignals(True)
        self.circle_form.set_choices("draw_on", [(n, n) for n in channels] or [("", "")])
        self.circle_form.blockSignals(False)
        previous = set(_checked(self.show_on_list))
        known = {self.show_on_list.item(i).text() for i in range(self.show_on_list.count())}
        self.show_on_list.blockSignals(True)
        self.show_on_list.clear()
        for name in channels:
            # New channels show the circle by default.
            self.show_on_list.addItem(_checkable(name, name in previous or name not in known))
        self.show_on_list.blockSignals(False)
        self._update_enabled()

    # --------------------------------------------------------------- display
    def _current_channel(self) -> str | None:
        item = self.channel_list.currentItem()
        return item.text() if item is not None else None

    def _channel_unit(self, name: str) -> str:
        if self.images and name in self.images[self.ref]:
            return self.images[self.ref][name].units["z"]
        return self.handles[self.ref].reader.units.get(name, "") if self.handles else ""

    def _show_style(self) -> None:
        name = self._current_channel()
        if name is None or name not in self.styles:
            self.style_form.setEnabled(False)
            return
        style = self.styles[name]
        self.style_label.setText(name)
        self.style_form.blockSignals(True)
        for key in ("cmap", "unit"):
            self.style_form.fields[key].blockSignals(True)
        self.style_form.set_choices("unit", [(label, label) for label, _ in
                                             jpk_model.unit_options(self._channel_unit(name))])
        self.style_form.set_value("unit", style.unit)
        self.style_form.set_value("cmap", style.cmap)
        for key in ("cmap", "unit"):
            self.style_form.fields[key].blockSignals(False)
        self.style_form.blockSignals(False)
        self.style_form.setEnabled(True)

    def _style_changed(self) -> None:
        name = self._current_channel()
        if name is None or name not in self.styles:
            return
        label = self.style_form.value("unit")
        scale = dict(jpk_model.unit_options(self._channel_unit(name))).get(label, 1.0)
        self.styles[name] = jpk_model.ChannelStyle(self.style_form.value("cmap"), label, scale)
        if name in self._plotted_channels():
            self._draw()

    def _unit_changed(self) -> None:
        self._sync_circle_fields()
        self._draw()

    def _region_changed(self) -> None:
        self.shift_form.set_row_visible("window", self.shift_form.value("region") == "circle")

    # ---------------------------------------------------------------- circle
    def _set_reference_circle(self, circle) -> None:
        if self.circles and model.same_circle(circle, self.circles[self.ref]):
            return
        self.circles = model.follow_reference(self.circles, self.ref, circle)
        self._circle_changed(reference=True)

    def _circle_changed(self, *, reference: bool) -> None:
        # Detection around the circle used the old reference circle: discard it.
        if reference and self._auto_settings and self._auto_settings[0] == "circle":
            self._clear_auto()
        self._build_overlays()

    def apply_circle_fields(self) -> None:
        if not self.circles:
            return
        v = self.circle_form.values()
        if v["radius"] <= 0:
            self.remove_circle()
            return
        self._set_reference_circle(jpk_model.CircleROI.from_unit(
            (v["x"], v["y"]), v["radius"], self.extent_unit))

    def remove_circle(self) -> None:
        if self.circles:
            self._set_reference_circle(None)

    def reset_circles(self) -> None:
        reference = self.circles[self.ref] if self.circles else None
        if reference is None:
            return
        self.circles = [jpk_model.CircleROI(reference.x, reference.y, reference.radius)
                        for _ in self.circles]
        self._circle_changed(reference=False)

    def _sync_circle_fields(self) -> None:
        circle = self.circles[self.ref] if self.circles else None
        if circle is None:
            return
        (cx, cy), r = circle.in_unit(self.extent_unit)
        self.circle_form.blockSignals(True)
        self.circle_form.set_values({"x": cx, "y": cy, "radius": r})
        self.circle_form.blockSignals(False)

    def _on_mouse(self, event) -> None:
        """Read the circles back after the user drew, moved or deleted one."""
        if event.name == "button_press_event" and event.button != 3:
            return
        if event.name == "button_release_event" and event.button != 1:
            return
        unit = self.extent_unit
        read = {}
        for i, selector in self._selectors.items():
            center, radius = selector.get_params()
            read[i] = (None if center is None
                       else jpk_model.CircleROI.from_unit(center, radius, unit))
        if self.ref in read and not model.same_circle(read[self.ref], self.circles[self.ref]):
            self._set_reference_circle(read[self.ref])
            return
        reference = self.circles[self.ref] if self.circles else None
        changed = False
        for i, circle in read.items():
            if i == self.ref or model.same_circle(circle, self.circles[i]):
                continue
            changed = True
            if circle is None or reference is None:
                # Right-click on another file: back to the reference position.
                circle = reference
            self.circles[i] = (None if circle is None else
                               jpk_model.CircleROI(circle.x, circle.y,
                                                   reference.radius if reference
                                                   else circle.radius))
        if changed:
            self._circle_changed(reference=False)

    # ----------------------------------------------------------------- shift
    def _clear_auto(self) -> None:
        self.auto_results = []
        self._auto_settings = None
        self.shift_form.set_choices("use_channel", [("", "")])

    def detect_shift(self) -> None:
        if len(self.images) < 2:
            return
        region = self.shift_form.value("region")
        window = float(self.shift_form.value("window"))
        circle = self.circles[self.ref] if region == "circle" else None
        if region == "circle" and circle is None:
            self.ctx.log.warning("Draw a circle on the reference first (or detect on the "
                                 "whole image).")
            return
        labels, images, ref = self.labels, self.images, self.ref
        channels = self._plotted_channels()

        def _done(results) -> None:
            if images is not self.images or ref != self.ref:
                return          # files, channels or reference changed meanwhile
            self.auto_results = results
            self._auto_settings = (region, circle)
            options = [(n, n) for n in dict.fromkeys(r.source for r in results)]
            self.shift_form.set_choices("use_channel", options or [("", "")])
            for r in results:
                if r.peak is not None and r.peak < 0.3:
                    self.ctx.log.warning(
                        f"AFM shift: weak correlation ({r.peak:.2f}) for {r.file}, "
                        f"{r.source}; that shift is unreliable.")
            self._update_tables()

        self.run_job("Detecting the shift between scans",
                     lambda progress: model.detect_shifts(labels, images, ref, channels,
                                                          circle=circle, window=window,
                                                          progress=progress),
                     _done, disable=[self.detect_button], cancellable=True)

    def apply_detected(self) -> None:
        channel = self.shift_form.value("use_channel")
        if not self.auto_results or not channel:
            return
        if self.circles[self.ref] is None:
            self.ctx.log.warning("Draw a circle on the reference first.")
            return
        self.circles = model.apply_detected(self.circles, self.ref, self.auto_results,
                                            channel)
        self._circle_changed(reference=False)

    def copy_shifts(self) -> None:
        if self.results:
            QApplication.clipboard().setText(
                model.shift_text(model.shift_rows(self.results, self.extent_unit),
                                 self.extent_unit))
            self.ctx.log.info("Shift table copied to the clipboard.")

    def copy_stats(self) -> None:
        if self.stats:
            QApplication.clipboard().setText(model.stats_text(self.stats))
            self.ctx.log.info("Circle statistics copied to the clipboard.")

    def _update_tables(self) -> None:
        self._sync_circle_fields()
        unit = self.extent_unit
        labels = self.labels
        manual = model.manual_shifts(labels, self.circles, self.ref) if self.images else []
        self.results = manual + list(self.auto_results)
        label = dict(jpk_model.EXTENT_UNITS)[unit]
        self.shift_table.setHorizontalHeaderLabels(
            ["file", "source", f"dx ({label})", f"dy ({label})", f"|d| ({label})",
             "correlation"])
        _fill_table(self.shift_table, model.shift_rows(self.results, unit), _SHIFT_COLUMNS)
        self.stats = model.stats_rows(labels, self.images, self.styles, self.circles, unit)
        _fill_table(self.stats_table, self.stats, _STAT_COLUMNS)
        self._update_enabled()
        self.plot.update_export_buttons()

    def _export_settings(self) -> dict:
        return {"channels": self._plotted_channels(), "process": self.process,
                "region": (self._auto_settings or ("image",))[0],
                "window": float(self.shift_form.value("window"))}

    def _export_payloads(self):
        if not self.handles or not self.images:
            return None
        return model.export_payloads(self.handles, self.ref, self.results, self.stats,
                                     self.circles, **self._export_settings())

    def _figure_info(self):
        if not self.handles:
            return None
        return model.figure_info(self.handles, self.ref, self.results, self.circles,
                                 **self._export_settings())

    # --------------------------------------------------------------- drawing
    def _detach(self) -> None:
        for selector in self._selectors.values():
            selector.disconnect()
            if selector.circle is not None:
                try:
                    selector.circle.remove()
                except (ValueError, NotImplementedError):
                    pass
        self._selectors = {}
        for patch in self._overlays:
            try:
                patch.remove()
            except (ValueError, NotImplementedError):
                pass
        self._overlays = []
        for cid in self._cids:
            self.plot.canvas.mpl_disconnect(cid)
        self._cids = []

    def _draw(self) -> None:
        self._detach()
        self._axes = {}
        panel = self.plot
        if not self.images:
            panel.show_message("Tick at least one channel to plot." if self.handles
                               else "Add .jpk files to start.")
            self._update_tables()
            return
        panel.clear()
        panel.figure.set_layout_engine("compressed")
        channels = self._plotted_channels()
        nrows, ncols = len(self.images), len(channels)
        unit = self.extent_unit
        unit_label = dict(jpk_model.EXTENT_UNITS)[unit]
        ranges = (model.shared_ranges(self.images, channels)
                  if self.channel_form.value("shared_range") else {})
        labels = self.labels
        for i, per_file in enumerate(self.images):
            row_label = labels[i] if len(labels[i]) <= 32 else labels[i][:29] + "…"
            if i == self.ref:
                row_label += " (ref.)"
            for j, name in enumerate(channels):
                ax = panel.figure.add_subplot(nrows, ncols, i * ncols + j + 1)
                style = self.styles[name]
                kwargs = {}
                if name in ranges:
                    kwargs["vmin"], kwargs["vmax"] = ranges[name]
                plot_image(per_file[name], cmap=jpk_model.resolve_cmap(style.cmap),
                           cbar_scale=style.scale, cbar_label=style.unit,
                           extent_unit=unit, title=name if i == 0 else "", ax=ax,
                           **kwargs)
                ax.set_xlabel(f"x ({unit_label})" if i == nrows - 1 else "")
                ax.set_ylabel(f"{row_label}\ny ({unit_label})" if j == 0
                              else f"y ({unit_label})", fontsize="small")
                self._axes[(i, name)] = ax
        self._build_overlays(redraw=False)
        panel.draw()

    def _build_overlays(self, redraw: bool = True) -> None:
        """Circles: draggable on the 'Draw on' channel, dashed copies, reference marks."""
        self._detach()
        channels = self._plotted_channels()
        if not self._axes or not channels:
            self._update_tables()
            return
        draw_on = self.circle_form.value("draw_on")
        if draw_on not in channels:
            draw_on = channels[0]
        show_on = set(_checked(self.show_on_list)) | {draw_on}
        unit = self.extent_unit
        reference = self.circles[self.ref] if self.circles else None
        mark_reference = self.circle_form.value("show_reference") and reference is not None
        for i in range(len(self.images)):
            circle = self.circles[i] if i < len(self.circles) else None
            ax = self._axes[(i, draw_on)]
            if circle is not None:
                center, radius = circle.in_unit(unit)
                self._selectors[i] = DraggableCircle(ax, center=center, radius=radius)
            elif i == self.ref:
                self._selectors[i] = DraggableCircle(ax)        # draw a new circle here
            for name in channels:
                if name not in show_on:
                    continue
                target = self._axes[(i, name)]
                if circle is not None and name != draw_on:
                    center, radius = circle.in_unit(unit)
                    self._add_patch(target, Circle(center, radius, edgecolor="red",
                                                   fill=False, linestyle="--",
                                                   linewidth=1.2))
                if mark_reference and i != self.ref:
                    center, radius = reference.in_unit(unit)
                    self._add_patch(target, Circle(center, radius,
                                                   edgecolor=_REFERENCE_COLOUR, fill=False,
                                                   linestyle=":", linewidth=1.4))
        # Registered after the selectors' own callbacks, so they run afterwards.
        canvas = self.plot.canvas
        self._cids = [canvas.mpl_connect("button_release_event", self._on_mouse),
                      canvas.mpl_connect("button_press_event", self._on_mouse)]
        self._update_tables()
        if redraw:
            canvas.draw_idle()

    def _add_patch(self, ax, patch) -> None:
        ax.add_patch(patch)
        self._overlays.append(patch)
