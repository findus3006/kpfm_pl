'''This file shows how to align KPFM and timetrace data with the files ptu_test1 and KPFM_test1 found in the Test data folder.
The relevant packages are imported, the file path is defined, the files loaded depending on their type and then correlated.
In the end the data is plotted. '''

import kpfm_pl as k
import matplotlib.pyplot as plt
from kpfm_pl.plotting import set_scaled_rcparams
from kpfm_pl.utils import feature_scale
from pathlib import Path # This allows us to not specify the full path to the test data folder

# Define file path. Select a .ptu and a .txt file that are to be correlated
script_dir = Path(__file__).parent
files = k.pick_files(script_dir / 'Test data', as_path=False)

# Iterate over files, load and get laser triggers
for file in files:
    if file.endswith('.ptu'): # If the file is a PTU file
        ptu = k.PTU(file, dark_counts=2.25)
        tt = ptu.time_trace(bintime=0.05, laser_pulse=False)        # Signal
        tt_triggers = ptu.laser_triggers()

    elif file.endswith('.txt'): # If the file is a Zurich file
        z = k.Zurich(file)
        stream_names = z.stream_names()
        kpfm = z.average(stream_names[0], bintime=0.05)                               # Signal
        z_triggers = z.laser_triggers(stream_names[1], reference=kpfm, method="slope")

# Correlate the data, set smoothing parameters
m = k.Match(kpfm, tt,
          kpfm_triggers=z_triggers,
          timetrace_triggers=tt_triggers,
          kpfm_smoothing=20, timetrace_smoothing=1)

aligned = m.to_dict()
print("Pearson:", m.pearson())
print("Best lag (s):", m.cross_correlation()["best_lag"])

k = m.kpfm_signal()
t = m.timetrace_signal()

# Plot params setup
set_scaled_rcparams((16,9))

# Plot on two axis
fig, ax = plt.subplots(figsize=(16,9))
ax2 = ax.twinx()
ax.plot(k["x"], k["y"], label=k["label"], c="red")
ax2.plot(t["x"], feature_scale(t["y"]), label=t["label"], c="blue")
ax.axvline(k["meta"]["triggers"]["left"], color="k", linestyle="--", label="Left laser trigger")
ax.axvline(k["meta"]["triggers"]["right"], color="k", linestyle="--", label="Right laser trigger")
ax.set_title("Correlation of KPFM and time trace")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude of KPFM signal (V)")
ax2.set_ylabel("Scaled and shifted time trace (arb. units)")
ax.grid(True)
fig.legend()
plt.show()