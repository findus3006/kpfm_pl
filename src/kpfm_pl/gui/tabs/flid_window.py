"""Pop-out window for one FLID map: zoom, pan and colour-scale controls.

:class:`FlidWindow` shows a single FLID :class:`~kpfm_pl.core.Image` in its
own top-level window with the full navigation toolbar (pan/zoom tools, wheel
zoom, middle-drag pan) and fields for the colour scale: minimum, maximum,
colormap and linear/logarithmic scaling. Figure and CSV export work as in the
main plots.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LogNorm, Normalize
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ...core import Image
from ..canvas import PlotPanel
from ..forms import ParamForm

__all__ = ["FlidWindow", "COLORMAPS"]

COLORMAPS = (("viridis", "Viridis"), ("magma", "Magma"), ("inferno", "Inferno"),
             ("plasma", "Plasma"), ("cividis", "Cividis"), ("turbo", "Turbo"),
             ("hot", "Hot"), ("gray", "Grey"))
SCALES = (("linear", "Linear"), ("log", "Logarithmic"))


class FlidWindow(QWidget):
    """One FLID map in its own window.

    Parameters
    ----------
    ctx :
        Application context (file dialogs and log of the export buttons).
    image :
        The FLID map (``PTU.flid``/``flid_map`` output).
    title :
        Window and plot title.
    ylabel :
        Label of the counts axis.
    export_provider :
        Callable returning the map's :class:`~kpfm_pl.gui.export.ExportPayload`.
    """

    def __init__(self, ctx, image: Image, title: str, ylabel: str,
                 export_provider=None, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(title)
        self.resize(900, 700)
        self.image = image
        self.title = title
        self.ylabel = ylabel

        self.panel = PlotPanel(ctx, figsize=(6, 6))
        self.panel.set_export_provider(export_provider)
        data = self._finite()
        top = float(data.max()) if data.size else 1.0
        f = self.form = ParamForm()
        f.add_optional_float("vmin", "Colour min", None, default=0.0, minimum=-1e12,
                             maximum=1e12, step=max(top / 20, 1e-6), decimals=4,
                             auto_text="data minimum",
                             tooltip="Lower end of the colour scale. Unchecked: the "
                                     "smallest value of the map (smallest positive value "
                                     "above 0.1 % of the maximum for a log scale).")
        f.add_optional_float("vmax", "Colour max", None, default=top, minimum=-1e12,
                             maximum=1e12, step=max(top / 20, 1e-6), decimals=4,
                             auto_text="data maximum",
                             tooltip="Upper end of the colour scale. Values above are "
                                     "shown in the top colour. Unchecked: the map maximum.")
        f.add_choice("cmap", "Colormap", COLORMAPS, "viridis", tooltip="Colour map.")
        f.add_choice("scale", "Scale", SCALES, "linear",
                     tooltip="Logarithmic: shows weak regions of the map next to the "
                             "strong peak (only positive values).")
        f.changed.connect(self.update_colours)
        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setStyleSheet("color: #b36b00;")
        hint = QLabel("Toolbar: pan/zoom tools and Home (reset view).\nWheel: zoom "
                      "(Shift: x only, Ctrl: y only). Middle-drag: pan.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(placeholder-text);")

        side = QVBoxLayout()
        side.addWidget(f)
        side.addWidget(self.message)
        side.addWidget(hint)
        side.addStretch(1)
        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setFixedWidth(280)
        layout = QHBoxLayout(self)
        layout.addWidget(self.panel, 1)
        layout.addWidget(side_widget)
        self.draw()

    def _finite(self) -> np.ndarray:
        data = np.asarray(self.image.data, dtype=float)
        return data[np.isfinite(data)]

    def draw(self) -> None:
        img = self.image
        panel = self.panel
        panel.clear()
        ax = self.ax = panel.figure.add_subplot()
        extent = (*img.meta["lifetime_range"], *img.meta["intensity_range"])
        self.im = ax.imshow(img.data, extent=extent, origin="lower", aspect="auto",
                            cmap=self.form.value("cmap"))
        ax.set_box_aspect(1)
        self.colorbar = panel.figure.colorbar(
            self.im, cax=ax.inset_axes([1.04, 0.0, 0.05, 1.0]), label=img.units["z"])
        ax.set_xlabel(f"Lifetime ({img.units['x']})")
        ax.set_ylabel(self.ylabel)
        ax.set_title(self.title)
        self.update_colours()
        panel.draw()

    def norm(self) -> Normalize | None:
        """The colour normalisation for the current fields (``None`` if invalid)."""
        v = self.form.values()
        data = self._finite()
        top = float(data.max()) if data.size else 1.0
        hi = v["vmax"] if v["vmax"] is not None else top
        if v["scale"] == "log":
            # Smoothing leaves tiny positive tails; keep the auto minimum visible.
            positive = data[data > max(top * 1e-3, 0.0)]
            lo = v["vmin"] if v["vmin"] is not None else (
                float(positive.min()) if positive.size else hi * 1e-3)
            if lo <= 0:
                self.message.setText("A logarithmic scale needs a positive minimum.")
                return None
            norm: Normalize = LogNorm(vmin=lo, vmax=hi, clip=True)
        else:
            lo = v["vmin"] if v["vmin"] is not None else (float(data.min()) if data.size else 0.0)
            norm = Normalize(vmin=lo, vmax=hi)
        if not lo < hi:
            self.message.setText("The colour minimum must be smaller than the maximum.")
            return None
        self.message.setText("")
        return norm

    def update_colours(self) -> None:
        if not hasattr(self, "im"):
            return
        norm = self.norm()
        if norm is None:
            return
        self.im.set_cmap(self.form.value("cmap"))
        self.im.set_norm(norm)
        self.colorbar.update_normal(self.im)
        self.panel.canvas.draw_idle()
