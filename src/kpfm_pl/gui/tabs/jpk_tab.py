"""AFM (JPK) tab: calibrated channels side by side with a circular ROI."""

from __future__ import annotations

import math

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
from ..models import jpk as model
from .base import BaseTab

__all__ = ["JPKTab"]

_STAT_COLUMNS = ("channel", "unit", "pixels", "mean", "std", "min", "max")


class JPKTab(BaseTab):
    """View AFM channels of one ``.jpk`` file and compare them inside a circle."""

    title = "AFM (JPK)"

    def __init__(self, ctx, parent=None) -> None:
        super().__init__(ctx, parent)
        self.images: dict = {}                      # channel -> Image (plotted)
        self.styles: dict[str, model.ChannelStyle] = {}
        self.process = True                          # flag the images were loaded with
        self.circle: model.CircleROI | None = None   # in metres
        self.stats: list[dict] = []
        self._axes: dict = {}                        # channel -> Axes
        self._selector: DraggableCircle | None = None
        self._cids: list[int] = []
        self._mirrors: list[Circle] = []

        self.add_hint(
            "1. Open a .jpk file. Height, KPFM (Precision 5) and photon-counter "
            "channels are shown by default.\n"
            "2. Tick the channels to show; click a channel name to change its "
            "colormap and colorbar unit.\n"
            "3. Draw a circle on one channel (left-drag on the image; drag inside to "
            "move, right-click to delete). The statistics inside it are listed for "
            "every channel.\nHover over a field for an explanation."
        )
        self._build_file_group()
        self._build_channel_group()
        self._build_display_group()
        self._build_circle_group()
        self._build_export_group()

        self.plot = PlotPanel(ctx, navigation="minimal", figsize=(10, 6))
        self.plot.set_export_provider(self._export_payload)
        self.plot.show_message("Open a .jpk file to start.")
        self.content_layout.addWidget(self.plot)
        self._set_enabled(False)

    # ------------------------------------------------------------- controls
    def _build_file_group(self) -> None:
        layout = self.add_group("File")
        self.file_label = QLabel("No JPK file loaded.")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)
        self.add_button(layout, "Open JPK file…", lambda: self.ctx.open_and_load("jpk"))

    def _build_channel_group(self) -> None:
        layout = self.add_group("Channels")
        self.channel_list = QListWidget()
        self.channel_list.setToolTip("Tick the channels to plot. Click a name to edit "
                                     "its display settings below.")
        self.channel_list.setMinimumHeight(180)
        self.channel_list.itemChanged.connect(lambda _: self.reload())
        self.channel_list.currentItemChanged.connect(lambda *_: self._show_style())
        layout.addWidget(self.channel_list)
        f = self.channel_form = ParamForm()
        f.add_bool("process", "Processing", True, text="line levelling",
                   tooltip="Line-level and zero height channels and convert "
                           "Precision 6 from volts to newtons.\nUnchecked: raw "
                           "calibrated data.")
        f.add_choice("extent_unit", "Axes unit", model.EXTENT_UNITS, "um",
                     tooltip="Length unit of the image axes and the circle.")
        f.fields["process"].toggled.connect(lambda _: self.reload())
        f.fields["extent_unit"].currentIndexChanged.connect(lambda _: self._unit_changed())
        layout.addWidget(f)

    def _build_display_group(self) -> None:
        layout = self.add_group("Display of selected channel")
        self.style_label = QLabel("Click a channel name above.")
        self.style_label.setWordWrap(True)
        layout.addWidget(self.style_label)
        f = self.style_form = ParamForm()
        f.add_choice("cmap", "Colormap", model.COLORMAPS, "viridis",
                     tooltip="Colour scale of this channel.")
        f.add_choice("unit", "Colorbar unit", [("", "")],
                     tooltip="Unit of the colorbar and of the statistics table.")
        f.changed.connect(self._style_changed)
        f.setEnabled(False)
        layout.addWidget(f)

    def _build_circle_group(self) -> None:
        layout = self.add_group("Circle (region of interest)")
        f = self.circle_form = ParamForm()
        f.add_choice("draw_on", "Draw on", [("", "")],
                     tooltip="Channel on which the circle is drawn and moved with the "
                             "mouse. It is mirrored on all other channels.")
        f.add_float("x", "Centre x", 0.0, minimum=-1e6, maximum=1e6, decimals=4,
                    tooltip="Circle centre in the axes unit.")
        f.add_float("y", "Centre y", 0.0, minimum=-1e6, maximum=1e6, decimals=4,
                    tooltip="Circle centre in the axes unit.")
        f.add_float("radius", "Radius", 0.1, minimum=0.0, maximum=1e6, decimals=4,
                    tooltip="Circle radius in the axes unit.")
        f.fields["draw_on"].currentIndexChanged.connect(lambda _: self._draw())
        layout.addWidget(f)
        row = QHBoxLayout()
        layout.addLayout(row)
        self.set_circle_button = self.add_button(
            row, "Set circle", self.apply_circle_fields,
            tooltip="Place the circle at the centre and radius entered above.")
        self.remove_circle_button = self.add_button(
            row, "Remove circle", self.remove_circle, tooltip="Delete the circle.")
        self.stats_table = QTableWidget(0, len(_STAT_COLUMNS))
        self.stats_table.setHorizontalHeaderLabels(list(_STAT_COLUMNS))
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setMinimumHeight(130)
        self.stats_table.setToolTip("Statistics of the pixels inside the circle, in each "
                                    "channel's colorbar unit.")
        layout.addWidget(self.stats_table)
        self.copy_button = self.add_button(
            layout, "Copy statistics", self.copy_stats,
            tooltip="Copy the table (tab-separated) to paste into Excel or Origin.")

    def _build_export_group(self) -> None:
        layout = self.add_group("Data export")
        f = self.export_form = ParamForm()
        f.add_choice("channel", "Channel for CSV", [("", "")],
                     tooltip="'Export data (CSV)' above the plot writes this channel as "
                             "a matrix in SI units (first row x, first column y).")
        layout.addWidget(f)

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (self.channel_list, self.channel_form, self.circle_form,
                       self.set_circle_button, self.remove_circle_button,
                       self.copy_button, self.export_form):
            widget.setEnabled(enabled)

    # ------------------------------------------------------------- state
    def _handle(self):
        return self.ctx.state.jpk

    def checked_channels(self) -> list[str]:
        items = (self.channel_list.item(i) for i in range(self.channel_list.count()))
        return [item.text() for item in items if item.checkState() == Qt.CheckState.Checked]

    def set_checked_channels(self, channels) -> None:
        """Tick exactly ``channels`` (and reload once)."""
        self.channel_list.blockSignals(True)
        for i in range(self.channel_list.count()):
            item = self.channel_list.item(i)
            item.setCheckState(Qt.CheckState.Checked if item.text() in channels
                               else Qt.CheckState.Unchecked)
        self.channel_list.blockSignals(False)
        self.reload()

    @property
    def extent_unit(self) -> str:
        return self.channel_form.value("extent_unit")

    def on_file_changed(self, kind: str, handle) -> None:
        if kind != "jpk":
            return
        self._detach_selector()
        self.images, self.styles, self.circle, self.stats = {}, {}, None, []
        self.stats_table.setRowCount(0)
        self.channel_list.blockSignals(True)
        self.channel_list.clear()
        if handle is None:
            self.channel_list.blockSignals(False)
            self.file_label.setText("No JPK file loaded.")
            self._set_enabled(False)
            self.plot.show_message("Open a .jpk file to start.")
            return
        reader = handle.reader
        names = reader.channel_names()
        selected = model.default_channels(names)
        for name in names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if name in selected
                               else Qt.CheckState.Unchecked)
            self.channel_list.addItem(item)
            self.styles[name] = model.default_style(name, reader.units.get(name, ""))
        self.channel_list.blockSignals(False)
        self.file_label.setText(f"{handle.name}\nScan {reader.dx * 1e6:.3g} × "
                                f"{reader.dy * 1e6:.3g} µm, "
                                f"{len(reader.x)} × {len(reader.y)} pixels")
        self._set_enabled(True)
        if selected:
            self.channel_list.setCurrentRow(names.index(selected[0]))
        self.reload()

    # -------------------------------------------------------------- actions
    def reload(self) -> None:
        handle = self._handle()
        if handle is None:
            return
        channels = self.checked_channels()
        process = self.channel_form.value("process")
        if not channels:
            self.images = {}
            self._refresh_choices()
            self._draw()
            return

        self.run_job(f"Reading channels of {handle.name}",
                     lambda progress: (model.load_images(handle, channels, process), process),
                     self._images_loaded, disable=[self.channel_list])

    def _images_loaded(self, value) -> None:
        images, self.process = value
        # Precision 6 is returned in newtons when processed: adapt the default style.
        for name, image in images.items():
            if image.units["z"] and self.styles[name].unit not in dict(
                    model.unit_options(image.units["z"])):
                self.styles[name] = model.default_style(name, image.units["z"])
        self.images = images
        self._refresh_choices()
        self._show_style()
        self._draw()

    def _refresh_choices(self) -> None:
        options = [(name, name) for name in self.images]
        for form, key in ((self.circle_form, "draw_on"), (self.export_form, "channel")):
            form.set_choices(key, options)

    def _current_channel(self) -> str | None:
        item = self.channel_list.currentItem()
        return item.text() if item is not None else None

    def _show_style(self) -> None:
        name = self._current_channel()
        if name is None or name not in self.styles:
            self.style_form.setEnabled(False)
            return
        style = self.styles[name]
        unit = self.images[name].units["z"] if name in self.images else \
            self._handle().reader.units.get(name, "")
        self.style_label.setText(name)
        self.style_form.blockSignals(True)
        self.style_form.set_choices("unit", [(label, label)
                                             for label, _ in model.unit_options(unit)])
        self.style_form.set_value("unit", style.unit)
        self.style_form.set_value("cmap", style.cmap)
        self.style_form.blockSignals(False)
        self.style_form.setEnabled(True)

    def _style_changed(self) -> None:
        name = self._current_channel()
        if name is None or name not in self.styles:
            return
        unit = self.images[name].units["z"] if name in self.images else \
            self._handle().reader.units.get(name, "")
        label = self.style_form.value("unit")
        scale = dict(model.unit_options(unit)).get(label, 1.0)
        self.styles[name] = model.ChannelStyle(self.style_form.value("cmap"), label, scale)
        if name in self.images:
            self._draw()

    def _unit_changed(self) -> None:
        self._sync_circle_fields()
        self._draw()

    def apply_circle_fields(self) -> None:
        v = self.circle_form.values()
        if v["radius"] <= 0:
            self.remove_circle()
            return
        self.circle = model.CircleROI.from_unit((v["x"], v["y"]), v["radius"],
                                                self.extent_unit)
        self._draw()

    def remove_circle(self) -> None:
        self.circle = None
        self._draw()

    def copy_stats(self) -> None:
        if self.stats:
            QApplication.clipboard().setText(
                model.stats_text(self.stats, self.circle, self.extent_unit))
            self.ctx.log.info("Circle statistics copied to the clipboard.")

    def _export_payload(self):
        handle = self._handle()
        name = self.export_form.value("channel")
        if handle is None or name not in self.images:
            return None
        return model.image_payload(handle, self.images[name], process=self.process,
                                   style=self.styles.get(name), circle=self.circle,
                                   stats=self.stats)

    # -------------------------------------------------------------- circle
    def _detach_selector(self) -> None:
        if self._selector is not None:
            self._selector.disconnect()
            self._selector = None
        for cid in self._cids:
            self.plot.canvas.mpl_disconnect(cid)
        self._cids = []

    def _circle_from_selector(self, event=None) -> None:
        """Read the circle back after the user drew, moved or deleted it."""
        if self._selector is None:
            return
        center, radius = self._selector.get_params()
        if center is None:
            if self.circle is None:
                return
            self.circle = None
        else:
            new = model.CircleROI.from_unit(center, radius, self.extent_unit)
            if new == self.circle:
                return
            self.circle = new
        self._sync_mirrors()
        self._update_stats()

    def _sync_mirrors(self) -> None:
        for patch in self._mirrors:
            patch.remove()
        self._mirrors = []
        if self.circle is not None:
            center, radius = self.circle.in_unit(self.extent_unit)
            draw_on = self.circle_form.value("draw_on")
            for name, ax in self._axes.items():
                if name != draw_on:
                    patch = Circle(center, radius, edgecolor="red", fill=False,
                                   linestyle="--", linewidth=1.2)
                    ax.add_patch(patch)
                    self._mirrors.append(patch)
        self.plot.canvas.draw_idle()

    def _sync_circle_fields(self) -> None:
        if self.circle is None:
            return
        (cx, cy), r = self.circle.in_unit(self.extent_unit)
        self.circle_form.blockSignals(True)
        self.circle_form.set_values({"x": cx, "y": cy, "radius": r})
        self.circle_form.blockSignals(False)

    def _update_stats(self) -> None:
        self._sync_circle_fields()
        self.stats = (model.circle_stats(self.images, self.styles, self.circle,
                                         self.extent_unit)
                      if self.circle is not None and self.images else [])
        self.stats_table.setRowCount(0)
        for row in self.stats:
            index = self.stats_table.rowCount()
            self.stats_table.insertRow(index)
            for col, key in enumerate(_STAT_COLUMNS):
                value = row[key]
                text = f"{value:.4g}" if isinstance(value, float) else str(value)
                self.stats_table.setItem(index, col, QTableWidgetItem(text))
        self.stats_table.resizeColumnsToContents()

    # -------------------------------------------------------------- drawing
    def _draw(self) -> None:
        self._detach_selector()
        self._axes = {}
        self._mirrors = []
        panel = self.plot
        if not self.images:
            panel.show_message("Tick at least one channel to plot."
                               if self._handle() is not None else "Open a .jpk file to start.")
            self._update_stats()
            return
        panel.clear()
        # "compressed" keeps colorbars as tall as the (square) images.
        panel.figure.set_layout_engine("compressed")
        n = len(self.images)
        ncols = min(n, 3)
        nrows = math.ceil(n / ncols)
        unit = self.extent_unit
        unit_label = dict(model.EXTENT_UNITS)[unit]
        for i, (name, image) in enumerate(self.images.items(), 1):
            ax = panel.figure.add_subplot(nrows, ncols, i)
            style = self.styles[name]
            plot_image(image, cmap=model.resolve_cmap(style.cmap), cbar_scale=style.scale,
                       cbar_label=style.unit, extent_unit=unit, title=name, ax=ax)
            ax.set_xlabel(f"x ({unit_label})")
            ax.set_ylabel(f"y ({unit_label})")
            self._axes[name] = ax

        draw_on = self.circle_form.value("draw_on")
        if draw_on not in self._axes:
            draw_on = next(iter(self._axes))
        ax = self._axes[draw_on]
        if self.circle is not None:
            center, radius = self.circle.in_unit(unit)
            self._selector = DraggableCircle(ax, center=center, radius=radius)
        else:
            self._selector = DraggableCircle(ax)
        # Registered after the selector's own callbacks, so they run afterwards.
        canvas = panel.canvas
        self._cids = [canvas.mpl_connect("button_release_event", self._circle_from_selector),
                      canvas.mpl_connect("button_press_event", self._circle_from_selector)]
        self._sync_mirrors()
        self._update_stats()
        panel.draw()
