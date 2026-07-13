'''This file shows how to read TCSPC data with the ptu_test2 file found in the Test data folder.
The relevant packages are imported, the file path is defined, the file loaded and then plotted. '''

import kpfm_pl as k
import matplotlib.pyplot as plt
from kpfm_pl.plotting import set_scaled_rcparams
from pathlib import Path # This allows us to not specify the full path to the test data folder
import numpy as np

# Defining the file path
script_dir = Path(__file__).parent
file = script_dir / 'Test data' / 'ptu_test2_LaserPulse(DR1_800nW_1200_ChargeAccu_000).ptu'

# Load and get laser triggers
ptu = k.PTU(file, dark_counts=2.25)
tt = ptu.time_trace(bintime=0.05, laser_pulse=True)        # Signal
laser_triggers = ptu.get_markers()

# Plot params setup
set_scaled_rcparams((16,9))

# Plotting the time trace
fig, ax = plt.subplots(figsize=(16,9))
ax.plot(tt["x"], tt["y"], label=tt["label"])
first = True
for trigger in laser_triggers:
    ax.axvline(trigger, color='k', linestyle='--', label='Trigger signal' if first else None)
    first = False
ax.set_ylabel("Counts")
ax.set_xlabel("Time (s)")
ax.set_title("Charge accumulation")
ax.grid(True)
ax.legend()
plt.show()

# Get the time trace data from 280 to 480 seconds
tt = ptu.time_trace(bintime=0.05, tstart=280, tend=480, laser_pulse=False)

# Get the top 70% of time bins (based on count level)
threshold_result = ptu.threshold(on=0.7)

# The "on" histogram contains the lifetime from the top 70% time bins
on_histogram = threshold_result["on"]
off_histogram = threshold_result["off"]

# Fit the lifetime on the selected ROI
fit_result = ptu.fit_lifetime(on_histogram, n_exp=2, mode="manual", start=0.1, end=70.0)

# Plot the result
fig, ax = plt.subplots(figsize=(16,9))
ax.plot(fit_result["shifted_histogram_x"], on_histogram.y, 'bo', label='Data', markersize=4, alpha=0.6)
ax.plot(fit_result["t_fit"], fit_result["curve"], 'r-', linewidth=2, label=fit_result["label"])
ax.set_ylabel("Counts")
ax.set_xlabel("Time (ns)")
ax.set_title("On state lifetime and fit")
ax.grid(True)
ax.legend()
plt.show()