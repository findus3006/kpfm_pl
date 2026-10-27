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
also used by *FLID* and *Correlate*.

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

Explore one `.ptu` file.

1. **Time trace**: photons per bin. Set the *bin time* and optionally a
   *start*/*end* time, then *Update time trace*. Red dashed lines are the laser
   triggers. For old recordings without laser markers, untick *Laser markers*
   (the trace is then centred and its two edges serve as triggers).
2. **On/off states**: bins above *On state* (% of the maximum) form the on
   state, bins below *Off state* the off state. The thresholds are drawn on
   the trace.
3. **Lifetime fit**: choose the histogram (whole trace, on or off state), the
   number of exponentials and how the fit window starts. In *Manual* mode you
   can drag across the histogram plot to set start and end. The results
   (amplitudes, lifetimes in ns, R²) appear in the table; the fitted curve and
   window are shown on the histogram. *Dark counts* are subtracted as an
   offset.
4. **Lifetime trace**: fits every time bin separately and plots the mean
   lifetime against time (grey: counts). This can take a while. Watch the
   progress bar; *Cancel* stops it.

Continuous-wave files have no lifetime information; the tab says so and
disables the analysis.

## FLID

Compare fluorescence-lifetime-intensity distributions of selected regions,
like `examples/example_flid.py`.

1. Open a `.ptu` file and check the *bin time* (default 0.5 s). The count
   ranges of the regions depend on it.
2. Draw regions (ROIs) on the time trace: **left-drag** draws a box, dragging
   inside moves it, the handles resize it, **Delete** removes the selected
   box. The list on the left shows every ROI. *Add ROI from numbers* adds an
   exact box. Without ROIs the window between the first and last laser
   trigger is used.
3. *Compute FLIDs*: one map per ROI, side by side, coloured like the ROI. All
   maps share the same lifetime range (default 0–20 ns) and count range, so
   their values are directly comparable. ROIs without photons are skipped
   (see the log).

**CSV export**: pick the ROI and whether to export its FLID map or the
per-bin fits (time, counts, lifetimes and amplitudes per bin).

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
   coincides in both files and resamples both signals onto one time grid. The
   *Aligned* plot shows the window between the first and last trigger by
   default; change *Window start/end* or drag across the plot to choose
   another window. The table lists the time shift, grid step and window.

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
gui/logcapture.py  print(), warnings and uncaught errors -> log panel
gui/models/        Qt-free workflow logic, one module per tab (unit-testable)
gui/tabs/          one widget per tab, built on tabs/base.py
```

Adding a workflow: write a Qt-free model (parameter dataclasses, the backend
calls, an `ExportPayload` builder), a `BaseTab` subclass using `ParamForm`,
`PlotPanel` and `run_job`, register it in `app.TAB_CLASSES`, and add tests
(`tests/test_gui_*.py`, offscreen via `pytest-qt`). See the GUI rules in
`AGENTS.md`, in particular the note on the PTU's cached time trace.
