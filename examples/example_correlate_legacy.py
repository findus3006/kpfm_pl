'''This file shows how to align KPFM and timetrace data with the files ptu_test1 and KPFM_test1 found in the Test data folder.
The files KPFM_test1_ and ptu_test1_ are legacy files to verify their functionality.
The relevant packages are imported, the file path is defined, the files loaded depending on their type and then correlated.
In the end the data is plotted. '''

import kpfm_pl as k
import matplotlib.pyplot as plt
from kpfm_pl.plotting import set_scaled_rcparams
from kpfm_pl.utils import feature_scale
from pathlib import Path # This allows us to not specify the full path to the test data folder

# Define file path. Select a .ptu and a .txt file that are to be correlated
script_dir = Path(__file__).parent
files = [str(script_dir / 'Test data' / 'ptu_test1_NoLaserPulse_(360s_400nW_532nm_000).ptu'), 
         str(script_dir / 'Test data' / 'KPFM_test1_(JV001_DR1_25Hz_AM13_1V_400nW_532nm_Scan1).txt')]

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
        # Legacy trigger stream drops at laser-on, so keep both flanks
        z_triggers = z.laser_triggers(stream_names[1], reference=kpfm, method="slope", flank="both")

# Align the data on the first laser trigger, set smoothing parameters
m = k.Match(kpfm, tt,
          kpfm_triggers=z_triggers,
          timetrace_triggers=tt_triggers,
          kpfm_smoothing=20, timetrace_smoothing=1)
print("Time shift (s):", m.offset)

# Extract the overlapping data between the first and last laser trigger.
# Alternatively use times on the shared grid, e.g. m.extract(10.0, 300.0)
window = m.extract(m.triggers[0], m.triggers[-1])
kp = window["kpfm"]
t = window["timetrace"]

# Plot params setup
set_scaled_rcparams((16,9))

# Plot on two axis
fig, ax = plt.subplots(figsize=(16,9))
ax2 = ax.twinx()
ax.plot(kp["x"], kp["y"], label=kp["label"], c="red")
ax2.plot(t["x"], feature_scale(t["y"]), label=t["label"], c="blue")
for i, point in enumerate(kp["meta"]["triggers"]):
    ax.axvline(point["time"], color="k", linestyle="--",
               label="Laser trigger" if i == 0 else None)
ax.set_title("Correlation of KPFM and time trace")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude of KPFM signal (V)")
ax2.set_ylabel("Scaled and shifted time trace (arb. units)")
ax.grid(True)
fig.legend()
plt.show()
