'''This file shows how to read timetrace data with the ptu_test2 file found in the Test data folder.'''

import kpfm_pl as k
import matplotlib.pyplot as plt
from pathlib import Path # This allows us to not specify the full path to the test data folder

script_dir = Path(__file__).parent
file = script_dir / 'Test data' / 'ptu_test2_LaserPulse(DR1_800nW_1200_ChargeAccu_000).ptu'

ptu = k.PTU(file, dark_counts=2.25)
tt = ptu.time_trace(bintime=0.05, laser_pulse=True)        # Signal
laser_triggers = ptu.get_markers()

fig, ax = plt.subplots(figsize=(16,9))
ax.plot(tt["x"], tt["y"], label=tt["label"])
for trigger in laser_triggers:
    ax.axvline(laser_triggers, color='k', linestyle='--', label='Trigger signal')
ax.set_ylabel("Counts")
ax.set_xlabel("Time (s)")
ax.set_title("Charge accumulation")
ax.grid(True)
ax.legend()
plt.show()