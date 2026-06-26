'''This file shows how to read AFM data with the jpk_test1 file found in the Test data folder.
The necessary imports are shown, the file path defined, the plotting dependencies imported, a circle
interactively drawn and the data plotted.'''

from kpfm_pl import JPK
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path # This allows us to not specify the full path to the test data folder

# Defining the file path
script_dir = Path(__file__).parent
file = script_dir / 'Test data' / 'jpk_test_(Drift1).jpk'

# Load and process
drift = JPK(file)
channels = drift.channel_names()
afm = drift.image("Height (measured)", process=True)
kpfm = drift.image("Precision 5")
photo = drift.image("Photon Counter 1")
jpk_data = drift.images(["Height (measured)", "Precision 5", "Photon Counter 1"]) # This returns a dictionary with the images

# Plot params setup
from kpfm_pl import plotting
plotting.set_scaled_rcparams((16, 9))

# Draw circle with proprietary plotting function
fig, ax, im, cbar = plotting.plot_image(afm, cmap=plotting.cmap_gold, cbar_scale=1e9, cbar_label="Height (nm)", extent_unit="um", title="Topography")
circle_params = plotting.make_circle(ax)

# Plot the data manually and add the circle as a patch
fig, ax = plt.subplots(figsize=(10,7))
im = ax.imshow(afm.data, extent=afm.extent_um, cmap=plotting.cmap_gold)
cbar = fig.colorbar(im)
plotting.set_cbar_formatter(cbar,afm.data, 1e9)
cbar.set_label("nm")
ax.set_title("AFM")
ax.add_patch(Circle(circle_params[0],circle_params[1], edgecolor = "red", fill=False))
fig, ax = plt.subplots(figsize=(10,7))
im = ax.imshow(kpfm.data, extent=kpfm.extent_um, cmap=plotting.cmap_spring)
cbar = fig.colorbar(im)
plotting.set_cbar_formatter(cbar,kpfm.data,1e3)
cbar.set_label("m" + kpfm.units["z"])
ax.set_title("KPFM")
ax.add_patch(Circle(circle_params[0],circle_params[1], edgecolor = "red", fill=False))
fig, ax = plt.subplots(figsize=(10,7))
im = ax.imshow(photo.data, extent=photo.extent_um, cmap='Reds')
cbar = fig.colorbar(im)
plotting.set_cbar_formatter(cbar,photo.data,1)
cbar.set_label(photo.units["z"])
ax.set_title("PL")
ax.add_patch(Circle(circle_params[0],circle_params[1], edgecolor = "red", fill=False))
plt.show()