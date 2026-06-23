"""Interactive rectangular region selector for matplotlib axes.

Optional GUI helper (requires matplotlib). Draw boxes by click-drag, move them by
dragging the interior, resize via the handles, delete the active box with
``Delete`` and press ``Enter`` to emit the saved coordinates. Use
:meth:`InteractiveBoxSelector.get_boxes` to retrieve the selections as dicts.
"""

from __future__ import annotations

from typing import Callable

__all__ = ["InteractiveBoxSelector"]


class InteractiveBoxSelector:
    """Draw, move and resize rectangular ROIs on a matplotlib ``Axes``.

    Parameters
    ----------
    ax :
        The axes to attach to.
    on_save :
        Optional callback invoked with the list of box dicts when ``Enter`` is
        pressed.
    box_color, box_alpha :
        Appearance of drawn boxes.
    handle_size :
        Resize-handle size in data units (auto-scaled when ``None``).
    """

    HANDLE_POSITIONS = [
        ("left", 0.0, 0.5), ("right", 1.0, 0.5),
        ("bottom", 0.5, 0.0), ("top", 0.5, 1.0),
        ("bottom_left", 0.0, 0.0), ("bottom_right", 1.0, 0.0),
        ("top_left", 0.0, 1.0), ("top_right", 1.0, 1.0),
    ]

    def __init__(self, ax, on_save: Callable | None = None, box_color: str = "royalblue",
                 box_alpha: float = 0.25, handle_size: float | None = None) -> None:
        self.ax = ax
        self.fig = ax.figure
        self.on_save = on_save
        self.box_color = box_color
        self.box_alpha = box_alpha
        self._handle_size = handle_size

        self.boxes: list[dict] = []
        self.active_idx: int | None = None
        self._mode: str | None = None
        self._drag_start = None
        self._resize_handle = None
        self._box_at_start = None
        self._preview_rect = None

        self._cids = [
            self.fig.canvas.mpl_connect("button_press_event", self._on_press),
            self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion),
            self.fig.canvas.mpl_connect("button_release_event", self._on_release),
            self.fig.canvas.mpl_connect("key_press_event", self._on_key),
        ]

    # ------------------------------------------------------------------ public
    def disconnect(self) -> None:
        """Detach all event listeners."""
        for cid in self._cids:
            self.fig.canvas.mpl_disconnect(cid)

    def get_boxes(self) -> list[dict]:
        """Return the current boxes as ``{x_min, x_max, y_min, y_max}`` dicts."""
        return [self._box_to_coords(b) for b in self.boxes]

    # ------------------------------------------------------------------ geometry
    def _handle_size_data(self) -> float:
        if self._handle_size is not None:
            return self._handle_size
        xlim = self.ax.get_xlim()
        return (xlim[1] - xlim[0]) * 0.005

    @staticmethod
    def _box_to_coords(b: dict) -> dict:
        return {
            "x_min": min(b["x0"], b["x0"] + b["width"]),
            "x_max": max(b["x0"], b["x0"] + b["width"]),
            "y_min": min(b["y0"], b["y0"] + b["height"]),
            "y_max": max(b["y0"], b["y0"] + b["height"]),
        }

    def _handle_centers(self, b: dict) -> dict:
        x0, y0, w, h = b["x0"], b["y0"], b["width"], b["height"]
        return {name: (x0 + fx * w, y0 + fy * h) for name, fx, fy in self.HANDLE_POSITIONS}

    def _hit_handle(self, b: dict, x: float, y: float):
        hs = self._handle_size_data()
        for name, (cx, cy) in self._handle_centers(b).items():
            if abs(x - cx) <= hs and abs(y - cy) <= hs:
                return name
        return None

    @staticmethod
    def _hit_interior(b: dict, x: float, y: float) -> bool:
        x0, y0, w, h = b["x0"], b["y0"], b["width"], b["height"]
        return min(x0, x0 + w) <= x <= max(x0, x0 + w) and min(y0, y0 + h) <= y <= max(y0, y0 + h)

    # ------------------------------------------------------------------ render
    def _draw_box(self, b: dict) -> None:
        import matplotlib.patches as patches

        x0, y0, w, h = b["x0"], b["y0"], b["width"], b["height"]
        is_active = self.active_idx is not None and b is self.boxes[self.active_idx]
        edge = "crimson" if is_active else self.box_color
        lw = 1.8 if is_active else 1.2

        if b.get("patch") is None:
            rect = patches.Rectangle((x0, y0), w, h, linewidth=lw, edgecolor=edge,
                                     facecolor=self.box_color, alpha=self.box_alpha, zorder=4)
            self.ax.add_patch(rect)
            b["patch"] = rect
        else:
            rect = b["patch"]
            rect.set_xy((x0, y0))
            rect.set_width(w)
            rect.set_height(h)
            rect.set_edgecolor(edge)
            rect.set_linewidth(lw)

        hs = self._handle_size_data()
        for i, (cx, cy) in enumerate(self._handle_centers(b).values()):
            if i >= len(b.get("handles", [])):
                hp = patches.Rectangle((cx - hs, cy - hs), 2 * hs, 2 * hs, linewidth=1,
                                       edgecolor=edge, facecolor="white", alpha=0.9, zorder=5)
                self.ax.add_patch(hp)
                b.setdefault("handles", []).append(hp)
            else:
                hp = b["handles"][i]
                hp.set_xy((cx - hs, cy - hs))
                hp.set_width(2 * hs)
                hp.set_height(2 * hs)
                hp.set_edgecolor(edge)

    def _redraw_all(self) -> None:
        for b in self.boxes:
            self._draw_box(b)
        self.fig.canvas.draw_idle()

    def _remove_box_patches(self, b: dict) -> None:
        if b.get("patch"):
            b["patch"].remove()
            b["patch"] = None
        for hp in b.get("handles", []):
            hp.remove()
        b["handles"] = []

    # ------------------------------------------------------------------ events
    def _on_press(self, event):
        import matplotlib.patches as patches

        if event.inaxes != self.ax:
            return
        if event.button == 3:  # right-click deselects
            if self.active_idx is not None:
                self.active_idx = None
                self._redraw_all()
            return
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return
        x, y = event.xdata, event.ydata

        for i in range(len(self.boxes) - 1, -1, -1):
            handle = self._hit_handle(self.boxes[i], x, y)
            if handle:
                self.active_idx, self._mode, self._resize_handle = i, "resize", handle
                self._drag_start = (x, y)
                self._box_at_start = {k: self.boxes[i][k] for k in ("x0", "y0", "width", "height")}
                self._redraw_all()
                return

        for i in range(len(self.boxes) - 1, -1, -1):
            if self._hit_interior(self.boxes[i], x, y):
                self.active_idx, self._mode = i, "drag"
                self._drag_start = (x, y)
                self._box_at_start = {k: self.boxes[i][k] for k in ("x0", "y0", "width", "height")}
                self._redraw_all()
                return

        self.active_idx, self._mode = None, "draw"
        self._drag_start = (x, y)
        if self._preview_rect:
            self._preview_rect.remove()
        self._preview_rect = patches.Rectangle((x, y), 0, 0, linewidth=1.5, edgecolor="crimson",
                                               facecolor=self.box_color, alpha=self.box_alpha,
                                               linestyle="--", zorder=4)
        self.ax.add_patch(self._preview_rect)
        self._redraw_all()

    def _on_motion(self, event):
        if event.inaxes != self.ax or self._mode is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - self._drag_start[0]
        dy = event.ydata - self._drag_start[1]

        if self._mode == "draw":
            x0, y0 = self._drag_start
            self._preview_rect.set_xy((min(x0, event.xdata), min(y0, event.ydata)))
            self._preview_rect.set_width(abs(dx))
            self._preview_rect.set_height(abs(dy))
            self.fig.canvas.draw_idle()
        elif self._mode == "drag" and self.active_idx is not None:
            b = self.boxes[self.active_idx]
            b["x0"] = self._box_at_start["x0"] + dx
            b["y0"] = self._box_at_start["y0"] + dy
            self._draw_box(b)
            self.fig.canvas.draw_idle()
        elif self._mode == "resize" and self.active_idx is not None:
            b, s, h = self.boxes[self.active_idx], self._box_at_start, self._resize_handle
            new_x0, new_y0, new_w, new_h = s["x0"], s["y0"], s["width"], s["height"]
            if "left" in h:
                new_x0, new_w = s["x0"] + dx, s["width"] - dx
            if "right" in h:
                new_w = s["width"] + dx
            if "bottom" in h:
                new_y0, new_h = s["y0"] + dy, s["height"] - dy
            if "top" in h:
                new_h = s["height"] + dy
            if abs(new_w) > 1e-10 and abs(new_h) > 1e-10:
                b["x0"], b["y0"], b["width"], b["height"] = new_x0, new_y0, new_w, new_h
                self._draw_box(b)
                self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if event.button != 1:
            return
        if self._mode == "draw":
            x0d, y0d = self._drag_start
            x1d = event.xdata if event.xdata is not None else x0d
            y1d = event.ydata if event.ydata is not None else y0d
            w, h = x1d - x0d, y1d - y0d
            if self._preview_rect:
                self._preview_rect.remove()
                self._preview_rect = None
            if abs(w) > 1e-10 and abs(h) > 1e-10:
                self.boxes.append({"x0": min(x0d, x1d), "y0": min(y0d, y1d),
                                   "width": abs(w), "height": abs(h),
                                   "patch": None, "handles": []})
                self.active_idx = len(self.boxes) - 1
                self._redraw_all()
        self._mode = self._drag_start = self._resize_handle = self._box_at_start = None

    def _on_key(self, event):
        if event.key == "delete" and self.active_idx is not None:
            self._remove_box_patches(self.boxes.pop(self.active_idx))
            self.active_idx = None
            self._redraw_all()
        elif event.key == "enter":
            coords = self.get_boxes()
            if self.on_save:
                self.on_save(coords)
