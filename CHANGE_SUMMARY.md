# Implemented code-review changes

Date: 2026-08-18

This change set implements the accepted findings in `CODE_REVIEW.md`, updates
the corresponding tests and documentation, and preserves the items marked by
the user to remain unchanged.

## PTU correctness and safety

- Kept legacy `laser_pulse=False` time traces and their photon matrices aligned
  through the same centring and padding transformation. The invariant
  `Signal.y == Signal.meta["matrix"].sum(axis=1)` now holds for current,
  legacy, and windowed traces.
- Defined time-window selection as a half-open interval and retained every
  acquisition bin that overlaps it, including the partial final bin. Returned
  x values now describe the actual selected bin-left times.
- Added per-bin durations and total contributing acquisition duration to trace
  and lifetime-histogram metadata. Thresholded on/off histograms carry their
  own duration, and lifetime fitting uses that provenance for dark-count
  correction instead of unrelated cached trace state.
- Changed FLID intensity to use raw photon totals before lifetime-curve
  normalisation. Lifetime-axis grouping now derives coordinates and data from
  the same boundaries, including odd final groups.
- Added separate bounded `lifetime_bins` and `intensity_bins` FLID parameters,
  plus validation for fit counts, grouping, empty selections, and histogram
  sizes.
- Hardened PTU header parsing with exact fixed-size reads and bounded,
  fully-consumed variable-length strings, float arrays, and binary blobs.
- Added `reference=` support to `PTU.laser_triggers()`. Trigger indices and
  times now refer to the same signal axis; out-of-window markers warn and
  return `None` instead of producing inconsistent point data.

## JPK loading and calibration

- Separated the unnamed preview page into `JPK.thumbnail`; it is no longer
  exposed as a malformed measurement channel.
- Validated measurement-channel dimensionality, declared TIFF dimensions,
  cross-channel shape consistency, scan dimensions, and numeric calibration
  values.
- Parsed cantilever sensitivity and spring constant into SI values.
- Fixed processed Precision 6 trace and retrace channels to perform numeric
  voltage-to-force conversion and report `N` as their z unit. Missing,
  malformed, or non-finite force calibration now raises a targeted error.

## Trigger and matching cohesion

- Added public `TriggerPoint`, `TriggerPair`, and `TriggerCollection` typed
  dictionaries and re-used them across PTU, Zurich, and Match.
- Reworked Zurich threshold detection around contiguous above-threshold runs.
  Each run contributes its first and last sample, fixing the previous shifted
  left edge. The `all` field is now a list of the same typed trigger points
  used by `left` and `right`.
- Standardised Match trigger metadata on the shared nested point format. Match
  indices refer to the common grid and satisfy
  `grid[point["index"]] == point["time"]`.
- Added early Match validation for signal type, finite strictly increasing time
  axes, sample count, smoothing/grid sizes, triggers, overlap, trigger-scaling
  indices/amplitudes, and statistical sample count/variance.
- Preserved input labels and units for identity scaling, used dimensionless
  units for feature/z-score scaling, and used the KPFM target unit for trigger
  scaling. Original signal context is retained under `meta["source"]`.

## Core containers and public API

- Enforced equal-length 1-D `Signal` arrays and 2-D `Image` data with matching
  coordinate-axis lengths.
- Enforced the documented unit-key sets and string unit values.
- Added standard dictionary-style iteration, `get()`, `items()`, and `values()`
  while preserving the compatible meaning of `len(Signal)` as sample count.
- Fixed image-extent conversion to scale x and y from their own source units.
- Refined the return-format promise: primary plottable products use `Signal`
  or `Image`; statistics, fits, raw events, and metadata retain their documented
  specialised return values.
- Made module public exports consistent at package level, including utility
  functions and lazy plotting/selector symbols, and added `__dir__()` so lazy
  names are discoverable without importing matplotlib during base import.

## Packaging, repository, examples, and documentation

- Moved the package version to the single source
  `src/kpfm_pl/_version.py`; package and wheel metadata now both report 1.3.0.
- Added and packaged the PEP 561 `py.typed` marker.
- Removed tracked egg metadata and Python bytecode. Added ignore rules for egg
  metadata, pytest cache, and wheel-build output (bytecode was already ignored).
- Corrected documentation and integration-test paths from the nonexistent
  `src/examples` / `tests/Test data` locations to `examples` and
  `examples/Test data`.
- Updated the current marker-based `example_correlate.py` and the separate
  `example_correlate_legacy.py` to use the standard nested Match trigger data.
- Updated the README and PTU, Zurich, JPK, and Match manuals for all accepted
  behavioural and API changes.

## Intentional compatibility notes

- Existing valid `Signal` and `Image` construction remains compatible, but
  malformed shapes or unit dictionaries now fail at construction time.
- Match-produced trigger metadata now uses the same nested point shape as the
  loaders; consumers must read `triggers["left"]["time"]` and
  `triggers["right"]["time"]`.
- Zurich's `triggers["all"]` is now a list of typed point dictionaries rather
  than a float-coercing NumPy matrix.
- For an explicit PTU lifetime matrix with non-zero dark counts, pass
  `bintime=` to `lifetime_histogram()` so acquisition-duration provenance is
  available.
- FLID defaults to a bounded 100 by 100 output histogram; callers can tune the
  two new histogram-size parameters independently.

## Items deliberately left unchanged

Per the user annotations in `CODE_REVIEW.md`, this change set does not alter:

- database file pairing or root-level classification;
- `io.py` measurement-file sorting;
- CI/version-matrix coverage or PTU record-type dispatch;
- any item under the review's **Additional improvements** section.

## Verification

- `pytest -q -p no:cacheprovider`: **170 passed**, with one expected warning
  from the no-marker legacy PTU test.
- `example_correlate.py`: completed against the new marker-based test data
  using the non-interactive matplotlib backend.
- `example_correlate_legacy.py`: completed against the legacy no-marker data
  with `laser_pulse=False`.
- `python -m compileall -q src/kpfm_pl examples tests`: passed.
- `python -m pip check`: no broken requirements.
- Fresh-process import check: version 1.3.0, lazy public names discoverable,
  and neither matplotlib nor tkinter imported by base `import kpfm_pl`.
- Isolated wheel build: produced `kpfm_pl-1.3.0-py3-none-any.whl`; verified
  that `kpfm_pl/py.typed` and versioned distribution metadata are included.
- `git diff --check`: passed.
