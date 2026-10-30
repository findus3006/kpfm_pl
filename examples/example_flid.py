'''This file shows how to select regions in a time trace and compare their FLIDs with the ptu_test2
file found in the Test data folder. The time trace is plotted, regions are drawn interactively as boxes
(time window x count range), a FLID is computed for every box and all FLIDs are plotted side by side.

Requires an interactive matplotlib backend (Qt when run as a script, `%matplotlib qt` or
`%matplotlib widget` in a notebook).'''

import kpfm_pl as k
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from kpfm_pl.plotting import set_scaled_rcparams, enable_zoom_pan
from kpfm_pl.selector import InteractiveBoxSelector
from pathlib import Path # This allows us to not specify the full path to the test data folder
import numpy as np

# Defining the file path
script_dir = Path(__file__).parent
file = script_dir / 'Test data' / 'ptu_test2_LaserPulse(DR1_800nW_1200_ChargeAccu_000).ptu'

# Load, build the time trace and get the laser triggers
bintime = 0.5
ptu = k.PTU(file, dark_counts=2.25)
tt = ptu.time_trace(bintime=bintime)        # Signal
laser_triggers = ptu.get_markers()
print(f"Laser markers found: {len(laser_triggers)}: \n {laser_triggers}")

# Plot params setup
set_scaled_rcparams((16,9))

# Plot the time trace and draw the regions of interest
fig, ax = plt.subplots(figsize=(16,9))
ax.plot(tt["x"], tt["y"], color="k")
for trigger in laser_triggers:
    ax.axvline(trigger, color="grey", linestyle="--")
ax.set_xlabel("Time (s)")
ax.set_ylabel(f"Counts/{bintime*1000:.0f}ms")
ax.set_title("Left-drag: draw ROI | Drag interior/handles: move/resize | Delete: remove selected ROI\n"
             "Wheel: zoom (Shift: x only, Ctrl: y only) | Middle-drag: pan | Close window when done")
ax.grid(True)
selector = InteractiveBoxSelector(ax)
enable_zoom_pan(ax)
plt.show(block=True)                        # Blocks until the window is closed

# Each box is a dict {"x_min", "x_max", "y_min", "y_max"} (time in s, counts), the ROI format of ptu.flid
rois = selector.get_boxes()
if not rois:
    # Nothing drawn: fall back to the window between the first and last laser pulse
    t_on, t_off = laser_triggers[0], laser_triggers[-1]
    # t_mid = (t_on + t_off) / 2
    rois = [{"x_min": t_on, "x_max": t_off, "y_min": 0, "y_max": tt["y"].max() + 1}]
    print("No ROI drawn. Using the window between the first and last laser pulse.")

# Compute one FLID per ROI (same bintime as the time trace, so the boxes' count ranges match).
# All FLIDs share the same histogram ranges, so their bins and occurrence values are comparable.
# Lifetimes or counts outside these ranges are dropped; widen them if your data extends further.
lifetime_range = (0, 20)                    # ns
intensity_range = (0, tt["y"].max() + 1)    # counts per time bin
flids = []
for i, roi in enumerate(rois):
    try:
        flids.append((i, roi, ptu.flid(roi=roi, bintime=bintime, n_exp=1,
                                       lifetime_range=lifetime_range, intensity_range=intensity_range)))
    except ValueError as err:               # e.g. a box containing no photons
        print(f"ROI {i + 1} skipped: {err}")
if not flids:
    raise SystemExit("None of the ROIs contained photons.")
# Sanity check without fitting: ptu.flid(..., method="mean_arrival") maps the mean photon arrival
# time per bin, comparable to the intensity-weighted mean of the fits. ptu.lifetime_trace(roi=roi, ...)
# followed by k.flid_map(...) redraws a map with other histogram settings without fitting again.
# flid warns when too few bins hold min_counts=100 photons and suggests a larger bintime.

# Plot the time trace with the ROIs on top and their FLIDs below
fig = plt.figure(figsize=(16,9), layout="constrained")
fig_tt, fig_flid = fig.subfigures(2, 1, height_ratios=[1, 1.4])

ax_tt = fig_tt.subplots()
ax_tt.plot(tt["x"], tt["y"], color="k", linewidth=0.8)
ax_tt.set_xlabel("Time (s)")
ax_tt.set_ylabel(f"Counts/{bintime*1000:.0f}ms")
ax_tt.set_title("Time trace and selected ROIs")
ax_tt.grid(True)

flid_axes = np.atleast_1d(fig_flid.subplots(1, len(flids), sharex=True, sharey=True))
for ax_flid, (i, roi, flid) in zip(flid_axes, flids):
    color = f"C{i}"
    ax_tt.add_patch(Rectangle((roi["x_min"], roi["y_min"]), roi["x_max"] - roi["x_min"],
                              roi["y_max"] - roi["y_min"], edgecolor=color, facecolor=color,
                              alpha=0.25, linewidth=2))
    # The histogram ranges are the exact image edges (flid.extent spans only the bin centres)
    extent = (*flid["meta"]["lifetime_range"], *flid["meta"]["intensity_range"])
    im = ax_flid.imshow(flid["data"], extent=extent, origin="lower", aspect="auto", cmap="viridis")
    fig_flid.colorbar(im, ax=ax_flid, label=flid["units"]["z"])
    ax_flid.set_title(f"ROI {i + 1}: {roi['x_min']:.0f}-{roi['x_max']:.0f} s", color=color)
    ax_flid.set_xlabel(f"Lifetime ({flid['units']['x']})")
flid_axes[0].set_ylabel(f"Counts/{bintime*1000:.0f}ms")
enable_zoom_pan(fig)
plt.show()
