'''This file shows how to read AFM data with the jpk_test1 file found in the Test data folder.'''

import kpfm_pl as k
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path # This allows us to not specify the full path to the test data folder

script_dir = Path(__file__).parent
file = script_dir / 'Test data' / 'jpk_test_(Drift1).jpk'

drift = k.JPK(file)
channels = drift.channel_names()
afm = drift.image("Height (measured)", process=True)
kpfm = drift.image("Precision 5")
photo = drift.image("Photon Counter 1")
jpk_data = drift.images(["Height (measured)", "Precision 5", "Photon Counter 1"])

fig, ax = plt.subplots(figsize=(10,7))
im = ax.imshow(afm.data*1e9, extent=afm.extent*np.array([1e6, 1e6, 1e6, 1e6]), cmap='afmhot')
cbar = fig.colorbar(im)
cbar.set_label("nm")
ax.set_title("AFM")
plt.show()