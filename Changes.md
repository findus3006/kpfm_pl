# New
## New features

# Bug

# Added in the past
## New features:
- [x] (v1.4.0) New **AFM shift** tab: up to 4 `.jpk` files, one row per file and one column per channel (channels common to all files). The circle is drawn on the reference file's *Draw on* channel and copied to the same scan position on every other file; *Show the circle also on* picks the channels with dashed copies, a blue dotted circle marks the reference position. Manual shift: drag each file's circle onto the feature → dx/dy/|d| table. Automatic shift: new backend helper `utils.image_shift` (FFT cross-correlation, sub-pixel, correlation coefficient as quality), on the whole image or a window around the circle; *Move circles to detected shift*. Circle statistics per file, CSV export (shifts + statistics) and figure metadata with all sources.
- [x] Count check for the bins passed into `flid()`: at least `min_counts` (default 100) photons in `coverage` (default 90 %, 50 % = median) of the bins, otherwise a warning with the suggested bin time per ROI/FLID. `auto_bintime=True` recomputes with that bin time (ROI count range scaled along). Also in `lifetime_trace()`/`arrival_time_trace()`. GUI: *Photons per time bin* in the FLID group, advice per ROI in the log, *Bin time: automatic (per ROI)* with the counts axis scaled per map.
- [x] FLIDs without fitting: `PTU.arrival_time_trace()` (mean photon arrival time per bin, dark counts subtracted, optional correction of the finite window) and `flid(method="mean_arrival")`. GUI: *Lifetime from* = Fit / Mean arrival time / Both (side by side on identical bins). Dark counts do matter (flat background pulls the mean towards T/2), so they are subtracted. Note: on the test file the 1-exp fit gives shorter lifetimes than the mean arrival time because the decay is multi-exponential (χ²ᵣ 4.1 for 1 exp vs. 1.06 for 2 exp).
- [x] The lifetime fit returns `tau_int` ± `tau_int_err` (intensity-weighted mean lifetime, error from the fit covariance; also `A_err`/`tau_err`), shown in the fit table and plot title.
- [x] FLIDs in the GUI are square; double-click a map (or *Pop out map(s)*) opens it in its own window with zoom/pan and colour min/max, colormap and lin/log.
- [x] Advanced fit settings: *Constraint* (`sum(A) ≤ 1`). It is only applied to normalised fits — the lifetime trace and FLID per-bin fits, and the lifetime fit with *Normalise* ticked — so raw-count fits are unaffected (there the constraint would cap the amplitudes at 1 count). Unset amplitude bounds follow the constraint (`A ≤ 1`).

## Bugs:
- [x] (v1.4.0) AFM tab: *Save figure* suggests `<file>_channels` (metadata lists all plotted channels) instead of the first channel's name.
- [x] (v1.4.0) AFM tab: *Export data (CSV)* writes every plotted channel, one file each (`<name>_<channel>.csv`); the *Channel for CSV* drop-down is gone.
- [x] Changing the map smoothing or the FLID axes no longer refits: `flid()` is now `lifetime_trace()` + the new `flid_map()`, and the GUI redraws the maps instantly from the stored per-bin lifetimes.
- [x] Fit progress counts the actual time bins of all ROIs instead of a fixed x/1000 per ROI.
- [x] The FLID has its own fit window (fit start, edge fraction, edge smoothing, start/end) plus *Copy fit window from 'Lifetime fit'*; the note explains that *Region* selects the time bins and the fit window the part of each decay.
- [x] ROI handles have a fixed size on screen (square, kept on zoom/pan/resize); whole box edges can be dragged and the cursor shows the action.
