# Graphical interface (`kpfm_pl.gui`)

A desktop application for analysing KPFM, photoluminescence (PTU) and AFM (JPK)
measurements without writing code. It uses the same functions as the Python
package, so its results match the scripts in `examples/`.

## Installing and starting

```bash
pip install -e .[gui]       # once: adds PySide6 and matplotlib
kpfm-pl-gui                 # start the program
python -m kpfm_pl.gui       # same, if the command above is not found
```

On Windows you can also double-click `KPFM-PL-GUI.bat` in the repository
folder; it runs the code in `src/` directly (dependencies still need
`pip install -e .[gui]`).

`kpfm-pl-gui` is created by `pip install`. If the command is not found
although the package imports, the installation predates the GUI: run
`pip install -e .[gui]` again (or use `python -m kpfm_pl.gui`).

## The window

- **Tabs** (top): one per workflow, described below.
- **Controls** (left of each tab): a short how-to, then the settings grouped
  by step. **Hover over any field** for an explanation of what it does.
- **Plots** (right): every plot has
  - the matplotlib toolbar (Home resets the view, the arrows step back and
    forward through views),
  - **mouse wheel** to zoom around the cursor (hold **Shift** for x only,
    **Ctrl** for y only) and **middle-button drag** to pan,
  - **Save figure…** (PNG, SVG or PDF) and **Export data (CSV)…**.
- **Log** (bottom): what the program did, plus warnings (orange) and errors
  (red). If something looks wrong, check here first.
- **Status bar**: the loaded files, the running computation with a progress
  bar, and a **Cancel** button for long fits.

Files are opened from the **File** menu (Ctrl+O PTU, Ctrl+K Zurich, Ctrl+J
JPK) or the *Open…* buttons in the tabs. One file of each kind is open at a
time, and all tabs share it: a PTU file opened in the *PL lifetime* tab is
also used by *Correlate*.

## Exported files and their metadata

Both export types record how the data was produced: analysis name, date,
`kpfm_pl` version, the source file(s), every parameter set in the tab and the
main results (fitted lifetimes, time shift, ROI bounds, statistics, …).

- **CSV**: starts with `# key: value` comment lines holding the metadata,
  then one header row with names and units, then the data. Empty cells are
  missing values (e.g. failed fits). Images (FLID maps, AFM channels) are
  written as a matrix: the first row holds x, the first column holds y.
  Reading it back: `pandas.read_csv(path, comment="#")`; Origin and Excel
  can skip the `#` lines on import.
- **Figures**: the metadata is stored as JSON inside the file (PNG text
  field "Description", SVG description, PDF subject).

## PL lifetime (PTU)

Everything about one `.ptu` file in one tab: time trace, regions of interest
(ROIs), lifetime fits, lifetime trace and FLID maps (like
`examples/example_ptu.py` and `examples/example_flid.py`). Click a group
title on the left to fold it away.

1. **Time trace**: photons per bin. Set the *bin time* and the window:
   *Start at* / *End at* is either a *Time* or a *Trigger*, chosen from a
   drop-down of all laser triggers of the file (`#2 – 482.048 s`, numbered as
   in the plot). The chosen trigger numbers are kept when you open the next
   file. Then *Update time trace*. Red dashed, numbered lines are the laser
   triggers. For old recordings without laser markers, untick *Laser markers*
   (the trace is then centred and its two edges serve as triggers).
2. **Regions of interest**: on the trace, **left-drag** draws a box, dragging
   inside moves it, the handles resize it, **Delete** removes the selected
   box. The list shows every ROI in its colour; *Add ROI from numbers* adds
   an exact box. An ROI is a time window *and* a range of counts per bin, so
   it belongs to the bin time it was drawn at: when you change the bin time,
   the count ranges are rescaled, and everything that uses ROIs runs with the
   trace's bin time.
3. **On/off states**: bins above *On state* (% of the maximum) form the on
   state, bins below *Off state* the off state. The thresholds are drawn on
   the trace.
4. **Lifetime fit**: *Histogram of* chooses the photons: whole trace, on or
   off state, one ROI, *All ROIs combined*, or *Each ROI separately* (one fit
   per ROI, overlaid in the ROI colours, one table column per ROI). Choose
   the number of exponentials and how the fit window starts; in *Manual*
   mode you can drag across the histogram plot to set start and end.
   *Dark counts* are subtracted as an offset.
   - **Fit quality**: the table lists the lifetimes, amplitudes and
     **χ²ᵣ** (reduced chi-squared with Poisson weights), also shown in the
     plot title. χ²ᵣ ≈ 1 (green) means the fit describes the histogram within
     photon noise; clearly larger values (orange) mean systematic deviations,
     e.g. one exponential too few. R² is listed too, but is close to 1 for
     almost any decay. The panel below the histogram shows the **weighted
     residuals** (counts − fit) / √counts, which should scatter around 0
     without structure.
   - **Advanced fit settings…**: initial guesses and lower/upper bounds of
     every lifetime τ (ns) and amplitude A (fraction of the histogram peak),
     **Poisson weighting** (recommended for TCSPC data: the low-count tail
     is no longer ignored) and robust losses against outliers. Ticked fields
     are used, unticked ones are automatic. The settings apply to the
     lifetime fit, the lifetime trace and the FLIDs; the line under the
     button summarises them.
5. **Lifetime trace**: fits every time bin separately and plots the mean
   lifetime against time (grey: counts). *Region* restricts it to one ROI
   (then with the trace's bin time). This can take a while; watch the
   progress bar, *Cancel* stops it.
6. **FLID**: *Compute FLIDs* builds fluorescence-lifetime-intensity maps with
   the trace's bin time (FLIDs usually need ≥ 0.1 s bins). The FLID's own
   *Region* (independent of the lifetime trace's) chooses what is mapped:
   *Each ROI* gives one map per ROI, side by side and coloured like the ROI
   (without ROIs: the window between the first and last laser trigger);
   *Whole time window* gives one map of the displayed trace; *ROI n* maps
   only that ROI. All maps share the same lifetime range (default 0–20 ns)
   and count range, so their values are directly comparable. ROIs without
   photons are skipped (see the log). For the
   **CSV export** pick the ROI and whether to export its map or the per-bin
   fits.

Exports: the time trace (with ROIs and trigger times), the fitted
histogram(s) with fit curve and weighted residuals, the lifetime trace and
the FLID maps, each with all parameters including the advanced fit settings.

Continuous-wave files have no lifetime information; the tab says so and
disables the analysis.

## Correlate (KPFM ↔ PL)

Align a Zurich lock-in export (`.txt`, KPFM signal and laser trigger) with a
PL time trace (`.ptu`) on their laser triggers, like
`examples/example_correlate.py`.

1. Open both files (or open a pair from the *Database* tab). For old files
   choose *Recording type: Legacy*, which reads triggers without PTU markers
   and from both flanks of the Zurich trigger signal.
2. **Find triggers** runs automatically: the *Triggers* plot shows the KPFM
   signal and the PL trace with their numbered triggers. The status line
   reports both counts and warns if they differ or none were found. Adjust
   streams, trigger method or flank if needed.
3. **Align and extract** shifts the PL trace so the chosen trigger
   coincides in both files and resamples both signals onto one time grid.
   *Align on trigger* is a drop-down of every trigger found in both files with
   its time in each (`#2 – Zurich 12.300 s / PTU 45.600 s`); it is filled by
   *Find triggers*. The *Aligned* plot shows the window between the first and
   last trigger by default; change *Window start/end* or drag across the plot
   to choose another window. A window bound set to *Trigger* offers a
   drop-down of the triggers on the aligned time axis (`#2 – 34.500 s`), so
   only existing triggers can be chosen. The table lists the time shift, grid
   step and window.

**CSV export** of the aligned plot contains the KPFM signal and the
(unscaled) PL trace on one shared time column. The 0–1 scaling is for
display only.

## AFM (JPK)

View calibrated AFM channels, like `examples/example_jpk.py`.

1. Open a `.jpk` file. Height, KPFM (Precision 5) and photon counter are
   shown by default; tick others in the channel list. *Processing: line
   levelling* removes a linear background from every scan line.
2. Click a channel name to change its colormap and colorbar unit.
3. **Circle**: on the channel chosen in *Draw on*, left-drag draws a circle,
   dragging inside moves it, a right-click deletes it. You can also type
   centre and radius. The circle is copied (dashed) onto every channel, and
   the table lists pixel count, mean, standard deviation, minimum and maximum
   inside it for each channel. *Copy statistics* puts the table on the
   clipboard for Excel or Origin.

**CSV export** writes the channel chosen in *Channel for CSV* in SI units
(metres, volts, …), including the circle and its statistics in the metadata.

## Database

Index a measurement archive and open measurement pairs from it.

- **Build from folder…** scans `<sample>/[<precipitation>/]<date>/` folders for
  `.txt`/`.ptu` pairs (see [database.md](database.md)). **Save** and **Open**
  store the index as JSON; **Export table (CSV)** writes the rows currently
  shown.
- **Filter** with the search box (every word must match somewhere) and the
  drop-downs (sample, wavelength, KPFM mode, …). Click a column header to
  sort.
- **Open selected pair** (or double-click a row) loads both files and switches
  to *Correlate*.
- **Add pair…** and **Remove selected** edit the index (never the files).
  Unsaved changes are marked, and you are asked before they are discarded.

## For developers

The GUI is an optional front end. The backend does not import it, and it uses
only the public `kpfm_pl` API.

```
gui/app.py         MainWindow, run()
gui/context.py     AppState (loaded files, shared by tabs) + AppContext (dialogs, loading)
gui/workers.py     JobRunner: background jobs with progress and cancel
gui/canvas.py      PlotPanel: matplotlib canvas + Save figure / Export CSV
gui/forms.py       ParamForm: labelled parameter fields
gui/export.py      CSV and figure export with metadata (Qt-free)
gui/fit_options.py Advanced fit settings dialog (FitOptions)
gui/logcapture.py  print(), warnings and uncaught errors -> log panel
gui/models/        Qt-free workflow logic, one module per tab (unit-testable)
gui/tabs/          one widget per tab, built on tabs/base.py (collapsible groups);
                   tabs/rois.py: RoiEditor (box selector + ROI list)
```

Adding a workflow: write a Qt-free model (parameter dataclasses, the backend
calls, an `ExportPayload` builder), a `BaseTab` subclass using `ParamForm`,
`PlotPanel` and `run_job`, register it in `app.TAB_CLASSES`, and add tests
(`tests/test_gui_*.py`, offscreen via `pytest-qt`). See the GUI rules in
`AGENTS.md`, in particular the note on the PTU's cached time trace.
