# AGENTS.md

Guidance for AI coding agents working in the `kpfm_pl` repository.

## What this project is

`kpfm_pl` is a small, well-structured Python toolbox for loading and
correlating data from combined **KPFM**, **photoluminescence** (PicoQuant PTU /
TCSPC) and **AFM** (JPK) experiments. It targets a research lab workflow, not a
general audience, so clarity and correctness matter more than feature breadth.

Design goals to preserve:

- **One package, three file types** – PTU (`.ptu`), Zurich lock-in (`.txt`),
  JPK AFM (`.jpk`).
- **Consistent return format** – every loader/analysis call returns a
  plotting-friendly `Signal` or `Image` (dict-like, with
  `x`/`y`/`label`/`units`/`meta`).
- **No plotting in the core** – data is returned ready for matplotlib.
  Plotting/matplotlib lives only in `plotting.py` and `selector.py` and is
  imported lazily so the base package stays headless-friendly.
- **Flexible getters** over hard-coded, side-effect-heavy methods.

## Repository layout

```
src/kpfm_pl/
    __init__.py     public API (+ lazy matplotlib/GUI attributes)
    core.py         Signal / Image containers (the common return format)
    ptu.py          PTU  – PicoQuant TTTR-T3 loader (time trace, lifetime, FLID)
    zurich.py       Zurich – lock-in .txt loader (KPFM, triggers)
    jpk.py          JPK    – AFM image loader (calibrated channels)
    match.py        Match  – align a PL time trace with a KPFM signal
    fitting.py      MultiExpFit + 2-D Gaussian helpers
    database.py     build/query a searchable databank of measurement pairs
    utils.py        pure signal-processing helpers (smooth, resample, ...)
    io.py           file dialogs + measurement-pair sorting (lazy tkinter)
    selector.py     optional interactive matplotlib ROI selectors
    plotting.py     optional plotting helpers (cmaps, colorbars, wheel zoom/pan, ROIs)
src/examples/       runnable scripts + a `Test data/` folder of sample files
tests/              pytest suite
docs/               per-module manuals (ptu, zurich, jpk, match, database, plotting)
```

## Environment & commands

- OS: Windows; shell: PowerShell. Do **not** use `cd` in commands – set the
  working directory instead.
- Python `>=3.10`.
- Install: `pip install -e .[dev]` (adds `pytest` and `matplotlib`).
- Runtime deps: `numpy`, `scipy`, `tifffile`. Matplotlib is optional (`plot` /
  `gui` / `dev` extras).
- Run tests: `pytest` (config in `pyproject.toml`, `testpaths = ["tests"]`).

## Conventions to follow

- **Return `Signal`/`Image`, not raw arrays**, from public loader/analysis
  methods. Populate `label`, `units` (`{"x","y"}` or `{"x","y","z"}`) and use
  `meta` for extra context (triggers, fit params, matrices, ...).
- **Triggers** are a single point `{"index", "time"}` (`utils.TriggerPoint`)
  or a chronological `list` of such points. Build them with
  `utils.trigger_points(times, axis)`, which snaps each time onto the axis so
  `axis[index] == time` always holds. Loaders return lists; `Match` consumes
  them. There is no left/right pair format any more.
- **`Match` only aligns and resamples** (shift + shared grid over all data).
  No amplitude scaling or statistics in the core – users do that in post on
  the output of `Match.extract`.
- **Keep the core headless.** Never add a top-level `import matplotlib`; put
  plotting behind lazy imports (see `__init__.__getattr__` and the local
  `import matplotlib.pyplot as plt` inside functions).
- **Pure helpers go in `utils.py`** (no I/O, no plotting) so they stay easy to
  test.
- **Public API** is declared in each module's `__all__` and re-exported from
  `src/kpfm_pl/__init__.py`; update both when adding public symbols.
- **Comments/docstrings**: this codebase deliberately keeps explanatory
  docstrings and a few inline comments. Do not strip them. Match the existing
  NumPy-style docstring format when adding new public functions.

## Testing discipline

- Add or update tests under `tests/` for any behavioural change. The suite
  covers `utils`, `fitting`, `core`, `plotting`, `selector`, the loaders
  (`ptu`, `zurich`, `jpk`), `match` and `database`. Interactive helpers are
  tested on the Agg backend by dispatching synthetic `MouseEvent`s.
- Prefer synthetic/in-memory fixtures over large binary files where possible.
- Do not weaken or delete existing tests without an explicit reason.

## Documentation

- User-facing docs live in `docs/` (one manual per module) and `README.md`.
- When you change a public API, update the relevant `docs/*.md`, the
  `README.md` quick-start/examples, and the example scripts in `src/examples/`
  if they use the changed call.
- The example scripts double as living documentation; keep them runnable
  against the files in `examples/Test data/`. A legacy script `example_correlate_legacy.py` 
  is provided to verify the functionality of the legacy files.

## Gotchas specific to this codebase

- **PTU `laser_pulse` flag** refers to the pulse generator in the laser driver
  (e.g. a 10 MHz laser emits a pulse every 100 ns, used as the PicoHarp sync).
  `True` is the default (and how data is recorded now): the pulse generator
  runs throughout the whole file regardless of whether the laser light is on,
  so laser triggers are read from the marker channel (`laser_triggers` /
  `get_markers`). `laser_pulse=False` is a legacy work-around for files with no
  marker: it centres/zero-pads the non-zero part of the trace and stores its
  two edges as a trigger list in `meta["triggers"]`. Bad practice, kept only
  for compatibility; `Match(...).extract(m.triggers[0], m.triggers[-1])`
  reproduces the legacy left/right window.
- **Zurich time axis** is exported negative and reversed – the parser flips it
  to forward time. Don't "fix" this again.
- **Zurich triggers vs. averaged signal**: pass `reference=<averaged Signal>`
  to `laser_triggers` so trigger indices line up with the binned axis used by
  `Match`.
- **Zurich flanks**: `laser_triggers` reports rising flanks only by default,
  matching the PicoHarp marker channel. Current trigger streams are short
  pulses (one rising flank per event). Legacy trigger streams are a level that
  *drops* at laser-on, so legacy files need `flank="both"`.
- **JPK images** are flipped vertically on load; only the thumbnail page
  carries the scan factors `dx`/`dy`. Channel calibration is keyed by a private
  TIFF tag (see `_CHANNEL_CALIBRATION`).
- **PTU files can be untrusted input**: string/blob lengths are validated
  against the remaining file size (`_read_length_prefixed`). Preserve these
  bounds checks.

## Workflow
- You may invoke subagents to handoff tasks to speed up work or to minimize context window size.
- Always ask the user to clarify ambiguity in features and prompts. Rather ask more than less.

## Scope & safety
- Do not create scratch files, notebooks or scripts in the repo root. Keep the
  workspace clean.