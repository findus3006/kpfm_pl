# Codebase review and proposed changes

Date: 2026-08-17

Scope: source modules, public API, tests, examples, documentation, packaging,
return-type cohesion, and targeted runtime checks against the bundled sample
files. No source code was changed as part of this review.

## Executive summary

The package has a good overall shape: responsibilities are separated cleanly,
the core import remains headless, the main data products usually use `Signal`
or `Image`, and the pure helpers are easy to test. The current test suite passes
on a modern stack, but the real-file PTU/JPK tests are skipped because they look
in the wrong directory. Manual checks against those files exposed several
scientific-correctness and data-cohesion problems that should be fixed before
adding features.

Highest-priority findings:

1. The legacy PTU `laser_pulse=False` path moves the time trace but not its
   backing photon matrix, so subsequent threshold/ROI analysis selects the
   wrong matrix rows.
2. FLID intensity is calculated after normalising every lifetime row, so its
   `counts` axis does not represent photon counts.
3. JPK `Precision 6` processing crashes because textual calibration values are
   multiplied without conversion to floats; its output unit would also remain
   `V` after conversion to force.
4. PTU lifetime dark-count correction uses the duration of the whole cached
   trace even when fitting an on/off or ROI subset.
5. PTU time-window bin labels, partial final bins, and cached trigger indices
   can disagree with the data they describe.
6. Zurich threshold triggers return an incorrect left edge for many ordinary
   rectangular pulses.
7. PTU variable-length array/blob tags are not both consumed and bounded
   correctly.
8. The real-file loader tests are all skipped in the current layout, hiding the
   issues above.

## Verification performed

- `pytest -q -p no:cacheprovider`: **104 passed, 16 skipped**.
- The 16 skips are the PTU/JPK real-file tests. They expect fixtures under
  `tests/Test data`, while the tracked files are under `examples/Test data`.
- `python -m compileall -q src/kpfm_pl examples tests`: passed.
- `pip check`: no broken installed requirements.
- Base import check: importing `kpfm_pl` did not import `matplotlib` or
  `tkinter`.
- Runtime used: Python 3.13.1, NumPy 2.2.2, SciPy 1.15.1, tifffile
  2025.10.16, matplotlib 3.10.0.
- Both bundled PTU files and the bundled JPK file can be opened on this stack.

This verifies one modern dependency combination only. The declared minimum
versions and all supported Python versions are not currently exercised by CI.

## Confirmed bugs and correctness risks

### P0/P1: fix before relying on derived scientific results

#### 1. Legacy PTU trace and matrix become misaligned

Locations: `src/kpfm_pl/ptu.py:239-263`, `src/kpfm_pl/ptu.py:295-304`,
`src/kpfm_pl/ptu.py:456-472`.

With `laser_pulse=False`, the non-zero part of `trace` is centred and padded,
but `sliced`/`matrix` is left in its original order. The returned
`Signal.meta["matrix"]` and cached matrix therefore no longer correspond row by
row to `Signal.y`. `get_roi_matrix()` and `threshold()` use the shifted trace as
a mask over the unshifted matrix.

Confirmed with the bundled no-pulse PTU: 693 of 720 bins differed between the
returned trace and `meta["matrix"].sum(axis=1)`; the trace's non-zero range was
24..695 while the matrix range was 0..671. The bundled `example_ptu.py` follows
this path and then calls `threshold()`, so its fitted histogram can be wrong.

Proposed change: apply the identical row transformation to the matrix, or keep
both arrays unshifted and represent the display/alignment offset explicitly.
Add an invariant test that `signal.y == signal.meta["matrix"].sum(axis=1)` for
both `laser_pulse` modes and for sliced windows.

#### 2. FLID “counts” lose the original intensity

Location: `src/kpfm_pl/ptu.py:396-408`.

Each lifetime row is divided by its maximum before `counts.append(sum(row))`.
That makes the intensity depend mainly on decay shape and bin count, not the
number of detected photons. This contradicts the documented FLID axes and can
change scientific interpretation.

Proposed change: capture `raw_counts = row.sum()` before normalisation and use
that as the FLID intensity. Keep normalisation only for fitting. Add a synthetic
test where two equal-shape rows with different amplitudes occupy different
count levels but recover the same lifetime.

#### 3. Dark-count correction is wrong for thresholded/ROI histograms

Location: `src/kpfm_pl/ptu.py:475-504`.

`fit_lifetime()` computes its offset from the size of the entire cached time
trace. A histogram returned by `threshold()` may contain only a small subset of
those time bins, yet it receives the full-duration dark-count estimate. An
explicit matrix or independently constructed `Signal` has the same hidden
dependency on whichever trace happens to be cached.

Proposed change: record the contributing time-bin count and bin width in the
histogram metadata, and derive the offset from that metadata. Alternatively,
accept an explicit acquisition duration/bin count. Reject a fit when the
required provenance is unavailable rather than silently using unrelated cache
state.

#### 4. `Precision 6` processing crashes and reports the wrong unit

Locations: `src/kpfm_pl/jpk.py:76-81`, `src/kpfm_pl/jpk.py:135-148`.

JPK metadata values are stored as strings. `_process()` multiplies the
sensitivity string by the spring-constant string, which raises `TypeError`.
This was reproduced with the bundled JPK file. If converted successfully, the
processed `Image` would still inherit `z="V"`, even though the method says it
converted voltage to force.

Proposed change: parse known numeric calibration fields once during load,
validate presence/finite values, perform numeric multiplication, and return a
processed unit of `N`. Add real-file tests for both trace and retrace Precision
6 channels and for missing/malformed calibration.

#### 5. PTU variable-length array/blob parsing is unsafe or desynchronised

Location: `src/kpfm_pl/ptu.py:101-141`.

`Float8Array` reads its byte-count field but does not consume the following
payload, so the next header tag is read from array data. `BinaryBlob` seeks by
the untrusted length without using the bounds check applied to strings. Negative
or oversized blob lengths are therefore not rejected at the point of use.

Proposed change: use one bounded length-prefixed payload reader for ANSI
strings, wide strings, float arrays, and blobs. Decode float arrays or skip
their validated payload deliberately. Expand the security tests to cover all
variable-length tag types and truncated fixed-size tag headers.

#### 6. FLID lifetime binning can create shape mismatches and excessive memory

Locations: `src/kpfm_pl/ptu.py:383-391`, `src/kpfm_pl/ptu.py:428-434`.

- For an odd lifetime-axis length and `bins > 1`, `reduceat()` can produce one
  more group than `np.arange(0, max_time, step)`, causing axis/mask mismatch.
- The histogram uses `bins=n_hist`, where `n_hist` is the number of valid time
  bins. Memory is therefore quadratic in the measurement duration; thousands
  of time bins can allocate a very large 2-D array.
- `bins <= 0`, empty ROIs, and invalid `n_exp` are not validated.

Proposed change: derive binned coordinates from the same group boundaries as
the reduced matrix, expose separate bounded `lifetime_bins` and
`intensity_bins` parameters, and validate empty/invalid inputs before fitting.

### P1/P2: behavioural bugs and silent data loss

#### 7. PTU time slicing can mislabel bins and discard a partial final bin

Location: `src/kpfm_pl/ptu.py:234-242`.

The data slice starts at `floor(tstart / bintime)`, but its axis starts at the
user's unaligned `tstart`. For example, `tstart=0.26`, `bintime=0.5` returns
the 0.0-second bin labelled as 0.26 seconds. `end_bin` also uses `floor`, so a
non-integral final acquisition bin is omitted.

Proposed change: define and document interval semantics, calculate bin edges
once, slice with `searchsorted`, and label bins from the actual selected edges
or centres. Use `ceil` where a partially covered final bin is intended.

#### 8. PTU trigger index/time pairs can disagree after a windowed trace

Location: `src/kpfm_pl/ptu.py:514-522`.

Hardware marker times are absolute, while indices are found on the most
recently cached time axis. If that trace is a subset or excludes a marker, the
index is clamped to the nearest endpoint but the returned time remains the
out-of-window marker time. The shared trigger invariant
`reference.x[index] == time` is then false.

Proposed change: accept an explicit reference `Signal`, as Zurich does, or
always map against a canonical full trace. Validate that both markers lie
inside the reference and either warn/return `None` or return time-only marker
metadata when no valid index exists.

#### 9. Zurich threshold detection does not return the pulse's left edge

Location: `src/kpfm_pl/zurich.py:155-174`.

The above-threshold indices are split in half and the maximum of each half is
used. For a pulse active at indices 2..5, this returns left=3 and right=5,
not 2 and 5. Multiple pulses are even less well defined.

Proposed change: for a single on-window use `on[0]` and `on[-1]`; for multiple
windows, find contiguous runs and expose their rising/falling edges. Make the
choice of first/last pulse explicit and share the same edge representation with
the slope method.

#### 10. Match accepts non-overlapping signals and creates a descending grid

Location: `src/kpfm_pl/match.py:152-187`.

There is no check that `t0 <= t1`. A synthetic non-overlap produced the grid
`[3, 2, 1]` with no paired finite samples; subsequent statistics fail or return
meaningless results. Empty signals, non-monotonic axes, invalid `n_points`, and
too-short statistical windows are also not checked.

Proposed change: validate `Signal` inputs, monotonic finite time axes, positive
smoothing/grid sizes, overlap, and the minimum number/variance of paired points.
Raise a targeted `ValueError` before interpolation or statistics.

#### 11. Database pairing can be arbitrary, and root-level files crash

Locations: `src/kpfm_pl/database.py:223-249`,
`src/kpfm_pl/database.py:272-323`, `src/kpfm_pl/database.py:365-381`.

- A `.txt`/`.ptu` pair directly under the supplied root produces
  `rel_parts == ()` and `_classify_folder()` raises `IndexError` (confirmed).
- The greedy matcher accepts a best score of zero, so unrelated files can be
  paired silently.
- Greedy, filename-order-dependent assignment need not maximise the overall
  quality when several candidates are present.

Proposed change: define root-level classification, require a minimum evidence
score (with an explicit single-pair fallback if desired), detect ties, and use
a deterministic maximum-weight assignment for multi-file folders. Store match
evidence/confidence in `Entry.meta` or a pairing report so ambiguous archival
decisions are auditable.

User: Leave as is. Not an issue.

#### 12. File sorting silently drops or misparses valid files

Location: `src/kpfm_pl/io.py:17`, `src/kpfm_pl/io.py:87-108`.

Grouping stores only one path per suffix, so two `.ptu` files with the same
wavelength/power overwrite one another. Decimal values are unsupported and can
be matched from the middle: `1.5nW` was parsed as `5nW`. Matching is also
case-sensitive and unmatched files are discarded by documented design.

Proposed change: reuse the database filename parser, anchor token boundaries,
support decimals/case variants, keep lists rather than one path per suffix, and
return unmatched/ambiguous files rather than losing them. A structured sort
result would be safer than a flattened tuple for laboratory data.

User: Leave as is. Not an issue.

#### 13. JPK exposes the thumbnail as a malformed data channel

Locations: `src/kpfm_pl/jpk.py:70-101`, `src/kpfm_pl/jpk.py:121-137`.

The bundled file exposes a channel named `"None"` with a 64x64 thumbnail, while
the final shared axes are length 128. `Image` accepts this mismatch, so plotting
or spatial analysis receives incoherent data.

Proposed change: treat the unnamed page strictly as metadata/thumbnail, not as
a calibrated measurement channel, or give it its own axes and explicit
thumbnail metadata. Validate every channel shape against the axes used to build
its `Image`.

## Datatype and API cohesion

### The “dict-like” containers do not enforce their documented contracts

Locations: `src/kpfm_pl/core.py:33-76`, `src/kpfm_pl/core.py:80-164`.

- `Signal` accepts scalar, multidimensional, unequal-length `x`/`y` values even
  though its docstring promises equal-length 1-D arrays.
- `Image` accepts non-2-D data and axes whose lengths do not match its rows and
  columns; the JPK thumbnail demonstrates the consequence.
- The classes support key indexing and `dict(obj)`, but are not
  `collections.abc.Mapping` objects and lack normal mapping methods such as
  `.get()`/`.items()`.
- `Image.extent_scaled()` applies the x-axis unit to both x and y, so mixed axis
  units are silently scaled incorrectly.

Proposed change: validate shapes and unit keys in `__post_init__`; either inherit
from `Mapping[str, Any]` and implement the full small mapping protocol, or
describe the containers only as supporting key access. Scale x/y independently.
Consider whether arrays should be copied/read-only to prevent metadata and data
from drifting after construction.

### Trigger data uses different shapes under the same name

Locations: `src/kpfm_pl/utils.py:291-303`, `src/kpfm_pl/zurich.py:221-234`,
`src/kpfm_pl/match.py:189-190`, `src/kpfm_pl/match.py:255-299`.

Loader triggers use the shared nested shape:

```python
{"left": {"index": int, "time": float},
 "right": {"index": int, "time": float}}
```

Zurich adds an `all` NumPy array whose integer index column is coerced to float.
Match then stores `meta["triggers"]` under the same name but flattens it to
`{"left": float, "right": float}`. Code cannot safely consume a `triggers`
value without knowing which component produced it.

Proposed change: keep the nested pair shape everywhere, using indices into the
common Match grid, and put additional flanks in a list of the same typed trigger
point shape. Define `TypedDict` aliases (or small dataclasses) for trigger point,
pair, and collection.

### Match hard-codes labels and units instead of preserving inputs

Location: `src/kpfm_pl/match.py:64-88`, `src/kpfm_pl/match.py:255-277`.

With `scaling="none"`, output units are always `V` and `counts` regardless of
the input signals. Labels are also replaced with generic names. This breaks
container metadata cohesion for valid alternative streams/units.

Proposed change: preserve input label/unit for identity scaling, derive
dimensionless units for normalisation, and make trigger-scaled units match the
actual target signal unit. Preserve source metadata under a namespaced key.

### The documented return-format promise is broader than the implementation

README and module docs say every loader/analysis call returns `Signal` or
`Image`, but public methods also return raw arrays, dicts, tuples, loader
objects, paths, statistics, and fit payloads. Examples include
`PTU.get_events`, `PTU.fit_lifetime`, `Zurich.laser_triggers`,
`fit_gaussian2d`, `Match.pearson`, and `JPK.get_save_path`.

Proposed change: refine the promise to “primary plottable data products return
`Signal`/`Image`” rather than forcing metadata/statistics into those containers.
For richer analysis results, introduce explicit `FitResult`,
`CorrelationResult`, and trigger types, or add backwards-compatible methods
that return a fitted `Signal`/`Image` with parameters in `meta`. Avoid a breaking
return-type migration inside the 1.x line.

### Public export surface is inconsistent

Location: `src/kpfm_pl/__init__.py:45-106`.

Lazy plotting helpers and selector classes can be imported from the package but
are absent from package `__all__`. Most utilities listed in
`kpfm_pl.utils.__all__` are not re-exported, while only `downsample_sum` is.

Proposed change: decide whether top-level re-export is required for all module
public symbols. Then make `__all__`, lazy loading, documentation, and tests
agree. Add `__dir__` for discoverability of lazy names.

## Compatibility, packaging, and repository hygiene

#### Version metadata disagrees

`pyproject.toml` and the checked-in egg metadata say 1.3.0, while
`src/kpfm_pl/__init__.py:42` says 1.2.4. Users and tools can report different
versions.

Proposed change: use one source of truth, preferably distribution metadata or a
single version file populated/read by the build.

#### `py.typed` is declared but absent

`pyproject.toml` includes `kpfm_pl = ["py.typed"]`, but no marker file exists.
Type checkers therefore cannot reliably treat the installed annotations as
supported package typing.

Proposed change: add and ship `src/kpfm_pl/py.typed` after tightening the public
annotations, or remove the package-data declaration until typing is supported.

#### Build products and bytecode are tracked

`src/kpfm_pl.egg-info/` and many Python 3.11/3.13/3.14 `.pyc` files are tracked,
despite `__pycache__/` being ignored. They create noisy diffs, can become stale,
and make the repository appear modified merely by importing/compiling it.

Proposed change: remove tracked bytecode and generated egg metadata from Git,
retain the ignore rules, and generate them only during local installation/build.

#### Documentation links point to a non-existent directory

README and all per-loader manuals link to `src/examples/...`, but examples are
stored under `examples/...`. These links are broken. The same path mismatch is
present in the repository guidance.

Proposed change: standardise on `examples/`, update links, and use a shared test
fixture path for integration tests.

#### Real-file integration coverage is accidentally disabled

Locations: `tests/test_ptu.py:9-13`, `tests/test_jpk.py:9-11`.

The skip markers look in `tests/Test data`; fixtures live in
`examples/Test data`. As a result, all loader integration tests report as skips
in a normal checkout. Existing JPK tests also do not cover Precision 6 or the
unnamed thumbnail, and no FLID tests exercise intensity correctness or binning.

Proposed change: centralise `SAMPLE_DATA_DIR` in `conftest.py`, fail CI when
required tracked fixtures are absent, and add the missing regression cases.
If large fixtures are optional, split “unit” and “integration” markers and make
CI run both explicitly.

User: Why so complicated? Why not simply update the path?

#### Compatibility testing is too narrow

The code works in the reviewed Python 3.13/latest-dependency environment, and
its syntax is compatible with the declared Python 3.10 minimum. However, there
is no visible CI matrix for Python 3.10 through current supported versions or
for minimum dependency versions. The PTU decoder also assumes one PicoHarp-style
32-bit T3 record layout without checking `TTResultFormat_TTTRRecType`; other
PicoQuant PTU record types could be silently misdecoded.

Proposed change: add CI for Python 3.10-current with both minimum and latest
dependencies. Validate the PTU record-type tag and either dispatch to a matching
decoder or reject unsupported types clearly.

User: Not necessary.

## Additional improvements

User: No need to do the additional improvements.

- `MultiExpFit` should validate non-empty equal-length 1-D `t`/`y`, `n_exp >= 1`,
  finite data, increasing time, weight shape, and sufficient samples before
  indexing `t[1]` (`src/kpfm_pl/fitting.py:62-73`, `140-153`).
- Weighted fits calculate residual sum of squares from weighted residuals but
  total sum of squares from unweighted `y`, so reported R-squared is not a
  standard weighted or unweighted metric (`fitting.py:171-184`). Define and
  document one metric.
- `PTU._build_matrix`, `Zurich.average`, utilities such as `bin_average` and
  `downsample_sum`, and plotting helpers need explicit positive-size and empty
  input validation rather than downstream NumPy exceptions.
- `plot_image()` does not set an origin (`plotting.py:155-156`) while JPK data is
  flipped on load and has ascending physical y coordinates. Establish and test
  one orientation convention; likely make `origin="lower"` explicit for
  coordinate-backed `Image` objects.
- `DraggableCircle.on_press()` does not restrict drawing/dragging to the left
  mouse button, while a separate handler uses right-click for deletion
  (`selector.py:366-420`). Right-drag can therefore enter drawing state. Add GUI
  event tests and explicit button checks.
- `JPK.get_save_path()` creates a directory despite being named like a pure
  getter (`jpk.py:151-155`). Either make creation explicit or rename/document
  the side effect.
- Library loaders and helpers print progress directly. Prefer `logging` with a
  `NullHandler`, or a `verbose`/callback option, so notebooks and batch jobs can
  control output.
- `example_jpk.py` indexes the result of `make_circle()` without handling its
  documented `None` result; cancelling the selection crashes the example.
- `example_correlate.py` relies on selecting exactly the right file types and
  can use undefined variables after cancel/incomplete selection. Validate the
  selected pair and avoid overwriting the imported module name `k`.

## Recommended implementation order

1. Repair the integration fixture path and add focused regression tests first.
2. Fix PTU trace/matrix alignment, time slicing, subset provenance/dark counts,
   FLID counts/binning, and variable-length tag safety.
3. Fix JPK numeric calibration/unit handling and exclude or separately model
   the thumbnail.
4. Fix Zurich threshold edges, Match overlap validation, database ambiguity,
   and file-sort data loss.
5. Enforce `Signal`/`Image` invariants and standardise trigger/unit metadata.
6. Align version/export/typing metadata; remove generated artifacts; repair
   documentation and examples.
7. Add Python/minimum-dependency CI and unsupported-format checks.

The first four steps can be kept largely backwards compatible. Container
validation may expose previously accepted invalid inputs, and any return-type
redesign should be staged with deprecations or reserved for a major release.
