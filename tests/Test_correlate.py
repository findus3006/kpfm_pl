'''This file shows how to align KPFM and timetrace data with the files ptu_test1 and KPFM_test1 found in the Test data folder.'''

import kpfm_pl as k
import matplotlib.pyplot as plt
from pathlib import Path # This allows us to not specify the full path to the test data folder

script_dir = Path(__file__).parent
files = k.pick_files(script_dir / 'Test data', as_path=False)


for file in files:
    if file.endswith('.ptu'): # If the file is a PTU file
        ptu = k.PTU(file, dark_counts=2.25)
        tt = ptu.time_trace(bintime=0.05, laser_pulse=False)        # Signal
        tt_triggers = ptu.laser_triggers()

    elif file.endswith('.txt'): # If the file is a Zurich file
        z = k.Zurich(file)
        stream_names = z.stream_names()
        kpfm = z.average(stream_names[0], bintime=0.05)                               # Signal
        z_triggers = z.laser_triggers(stream_names[1], reference=kpfm, slope_method=True)

m = k.Match(kpfm, tt,
          kpfm_triggers=z_triggers,
          timetrace_triggers=tt_triggers,
          kpfm_smoothing=20, timetrace_smoothing=1)

aligned = m.to_dict()
print("Pearson:", m.pearson())
print("Best lag (s):", m.cross_correlation()["best_lag"])

k = m.kpfm_signal()
t = m.timetrace_signal()
fig, ax = plt.subplots(figsize=(16,9))
ax.plot(t["x"], t["y"], label=t["label"])
ax.plot(k["x"], k["y"], label=k["label"])
ax.axvline(z_triggers["left"]["time"], color='k', linestyle='--', label='Left laser trigger')
ax.axvline(z_triggers["right"]["time"], color='k', linestyle='--', label='Right laser trigger')
ax.set_ylabel("Amplitude (V)")
ax.set_xlabel("Time (s)")
ax.set_title("Correlation between KPFM signal and Timetrace")
ax.grid(True)
ax.legend()
plt.show()