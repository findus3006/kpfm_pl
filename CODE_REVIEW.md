# Code review — proposed changes and bugs

Full-codebase review (compatibility, cohesion, general quality). No code has been
changed; every item below is a proposal. Line numbers refer to the current state
of the repository (v1.3.0).

Legend: **[bug]** wrong behaviour · **[consistency]** violates a package convention ·
**[compat]** portability/packaging/version issue · **[style]** cleanliness/API polish

---

## 1. High priority — wrong results or crashes on realistic input

### 1.1 `Zurich` default stream names no longer match real exports — [bug]
`src/kpfm_pl/zurich.py:27-28` define `KPFM_STREAM = "Demod 6 Sample Auxiliary Input 1"` /
`TRIGGER_STREAM = "... Input 2"`, but current exports (verified against
`examples/Test data/test3_...txt`) name the streams `"Demod 6 Sample Aux In 1"` / `"... 2"`.
So `z.signal()`, `z.average()` and `z.laser_triggers()` **with default arguments raise a bare
`KeyError`** on current data. The examples dodge this by indexing `stream_names()[0]/[1]`.
**Fix:** update the constants; better, resolve defaults tolerantly (match on suffix "Aux In 1"
or fall back to first/second stream with a warning) and raise a helpful error listing the
available streams (see 4.9). `docs/zurich.md` documents the stale names too.

User: This is no bug. Most files have that name.

### 1.2 `PTU.get_roi_matrix` silently applies the legacy centre/zero-pad transform — [bug]
`src/kpfm_pl/ptu.py:373` calls `time_trace(..., laser_pulse=False)`. That legacy path
*centres the non-zero part of the trace and zero-pads it* (`ptu.py:279-304`), so the ROI
matrix rows are shifted/padded relative to real time, and zero-padded rows pass a
`y_min <= 0` filter. It also injects fabricated `"triggers"` into `self._tt_cache`
(`ptu.py:310-311`), so a later `laser_triggers()` on a marker-less file returns trace-edge
triggers the user never asked for. `docs/ptu.md` describes the method as a plain filter.
**Fix:** use the default `laser_pulse=True`; unify the y-filter with `flid` (see 3.1).

User: I set `laser_pulse=True`.

### 1.3 `Match.cross_correlation` lag sign is documented backwards — [bug]
`src/kpfm_pl/match.py:466,472-481` + `docs/match.md`. With `corr = np.correlate(k, t, "full")`
and `lags = np.arange(-len(t)+1, len(k)) * dt`, the peak at positive lag means the **KPFM
signal lags the time trace** — the docstring and manual claim the opposite. The test suite
only checks the symmetric zero-lag case, so the sign is unverified. This flips the physical
interpretation of the reported `best_lag`.
**Fix:** correct the docstring + `docs/match.md` (or swap operands so the docs become true),
and add a test with a deliberately shifted pair asserting the sign.

### 1.4 `Match` scaling `"feature"`/`"zscore"` is broken with a single float trigger — [bug]
`src/kpfm_pl/match.py:348-353`. With a float trigger, `k_left == k_right`, so the window mask
`(k_t >= k_left) & (k_t <= k_right)` only matches samples *exactly equal* to the trigger time —
on real axes it's empty and the constructor raises, even though the class docstring, the
`"trigger"`-branch error message and `docs/match.md` all advertise these modes for float
triggers. Tests pass only because `np.linspace(0, 10, 101)` contains 2.0 exactly.
**Fix:** reuse the `_window_mask` single-trigger fallback (full overlap) in `_compute_scaling`;
add a test with an off-grid trigger time.

User: This is important. If method="trigger" is behaves like "none" but checking would be better. 
On this account, I need you to change the `laser_triggers()` method to work with only one trigger in the marker channel.
This is only required so the `_Trigger_Spec` can be used with the `Match` class. Print a warning that only one trigger was found.

### 1.5 `build_database` crashes when data files sit directly in the scan root — [bug]
`src/kpfm_pl/database.py:229` (`rel_parts[0]`), reached via `:377-378` and `add_entry:416-417`.
`_find_leaf_folders` includes `root` itself, so a flat folder gives `rel_parts == ()` and an
`IndexError`. Untested (fixtures always use sub-folders).
**Fix:** handle the empty tuple (use `root.name` as sample); validate `root` is a directory
with the same message style as `io.pick_files`. Add a flat-folder fixture.

### 1.6 `plotting.plot_image` renders JPK images vertically mirrored — [bug]
`src/kpfm_pl/plotting.py:155-156`. JPK data is flipped on load so row 0 = bottom; the
documented manual path (`README.md`, `docs/jpk.md`) uses `imshow(..., origin="lower")`, but
`plot_image` uses matplotlib's default `origin="upper"` — the helper and the docs disagree,
and the helper shows AFM images upside down.
**Fix:** `imshow_kw.setdefault("origin", "lower")`; assert `im.origin == "lower"` in
`tests/test_plotting.py`.

User: Using this function I could not see the flipped image you are describing. Looks fine to me.

### 1.7 Zurich time-axis flip is only correct for a uniform grid ending at 0.0 — [bug, latent]
`src/kpfm_pl/zurich.py:93-96`: `t = np.array([-p[0] ...])[::-1]` reverses **t but not v**.
This pairs each value with the right forward time *only* when the exported axis is uniformly
spaced and its last sample is exactly `0.0`. Any end offset or jitter silently mis-pairs
samples (error grows toward the middle of the record). Note: this proposal **keeps** the
documented negate-and-flip semantics (per AGENTS.md); it only replaces the fragile index
gymnastics with the robust equivalent.
**Fix:** `t = t_raw - t_raw[0]` (or negate **and** reverse both `t` and `v`); pin with a test
using a non-zero-terminated axis.

### 1.8 `PTU.time_trace` returns an unsigned integer `y` — [bug]
`src/kpfm_pl/ptu.py:221,269,312`. The matrix is `uint32`, so `Signal.y` is unsigned; any
downstream subtraction (`tt.y - baseline`, `np.diff`) wraps to ~4.29e9 instead of going
negative. Every other `Signal` in the package carries float data.
**Fix:** `trace = sliced.sum(axis=1, dtype=np.int64).astype(float)` (keep `meta["matrix"]`
integer for memory).

### 1.9 `utils.set_zero` NaN-poisons whole images; its `abs` is dead code — [bug]
`src/kpfm_pl/utils.py:290-293`. `data.min()` is NaN if a single pixel is NaN (dropped AFM
scan lines), wiping the whole image; and after subtracting the minimum `np.abs` is a no-op
that contradicts the docstring ("shift" is not "abs").
**Fix:** `return data - np.nanmin(data)`; add a NaN-bearing test. Related: `linear_line_level`
(`utils.py:270-287`) also NaNs an entire row if one pixel is NaN — mask non-finite samples in
the per-row fit (and consider vectorising with `np.linalg.lstsq`, dropping legacy `np.polyfit`).

User: Nan values will not be a problem. The source data will always be finite.

### 1.10 `PTU.threshold` truncates the threshold to an integer — [bug]
`src/kpfm_pl/ptu.py:566-567`: `trace >= int(on * max_counts)` floors the *threshold*
(`on=0.75`, max 10 → threshold 7 instead of 7.5), biasing on/off classification both ways.
The existing test uses values where flooring is a no-op.
**Fix:** drop both `int(...)` casts; add a test with a non-integral threshold.

User: This is not a bug but the intended behaviour.

### 1.11 `PTU._build_matrix` can raise a bare `IndexError` — [bug]
`src/kpfm_pl/ptu.py:220-225`. `n_tt` is sized from the header acquisition time, but `tt_idx`
comes from photon syncs; with `smooth_rep_rate=True` (rounded rep time) or files whose records
run marginally past the declared duration, the last photons index past the matrix end and
`np.add.at` raises with no context.
**Fix:** size from the data too: `n_tt = max(n_tt, int(tt_idx.max()) + 1)` (guard empty
events), or clip with a warning.

User: When changing this, add a comment in place explaining why this is done.

### 1.12 Continuous-wave PTU files yield a half-constructed object — [bug]
`src/kpfm_pl/ptu.py:69-72,95-100`. When `MeasDesc_GlobalResolution` is inf, `__init__`
returns early: `rep_time`, `resolution`, `acquisition_time`, `n_lt_bins`, `lt_time` are never
set and every public method dies with `AttributeError`. `docs/ptu.md` advertises the
attributes unconditionally.
**Fix:** set an explicit `is_cw` flag + safe defaults; raise a clear `ValueError` from the
binning methods; document the CW case.

### 1.13 `database._pair_leaf_files` pairs files that share nothing — [bug]
`src/kpfm_pl/database.py:307-323`. `best_score` starts at `-1`, so a score of 0 (no shared
DR / frequency / power / wavelength) still pairs the `.txt` with the first remaining `.ptu`.
The "No .ptu match found" warning can only fire on count mismatch, never on genuinely
unrelated files. Greedy order can also mis-assign on ties.
**Fix:** require `best_score > 0` (warn + skip otherwise); warn on ties. For robustness,
`scipy.optimize.linear_sum_assignment` over the score matrix is a drop-in global matcher.

### 1.14 `MultiExpFit.quality()["r_squared"]` is wrong for weighted fits — [bug]
`src/kpfm_pl/fitting.py:171-184`. `result.fun` holds *weighted* residuals but `ss_tot` uses
unweighted `y`; with any non-unit `weights` the reported r² is meaningless (can exceed 1).
The weighted array is also returned as `"residuals"` undocumented.
**Fix:** compute the metric from unweighted residuals (`self.curve() - self.y`) or report a
properly weighted r²; document which one `"residuals"` is.

### 1.15 `JPK.images(channels=[])` returns *all* channels — [bug]
`src/kpfm_pl/jpk.py:252`: `names = channels if channels else self.channel_names()` — an empty
list is falsy and silently means "everything", the opposite of what was asked.
**Fix:** `names = self.channel_names() if channels is None else channels`.

User: This is the intended behaviour. An empty list should return all channels.

---

## 2. Medium priority — edge-case crashes & robustness (incl. untrusted PTU input)

### 2.1 PTU header robustness (untrusted input) — [bug/robustness]
- **Record type never validated** (`ptu.py:163-204`): the PicoHarp-T3 bit layout is
  hard-coded; a HydraHarp/MultiHarp T3 or any T2 file decodes into plausible-looking but
  wrong photon times with no warning. Check `TTResultFormat_TTTRRecType` and raise for
  anything but PicoHarp T3 (`0x00010303`).
- **Missing required tags raise raw `KeyError`** (`ptu.py:93-100`); a string-typed tag makes
  `np.isinf` raise `TypeError`. Add a `_require_tag(name)` helper raising `ValueError` in the
  module's established style.
- **Degenerate timing values** (`ptu.py:97-98,167`): `GlobalResolution == 0` →
  `log10(0)`/`OverflowError`; rep time > 1 s → `round(x, -1)` can yield `0.0` → later
  ZeroDivision. Validate both resolutions are finite and > 0 after header parse.
- **`TDateTime`** (`ptu.py:141-142`): NaN/huge/negative values raise
  `ValueError`/`OSError` from `time.gmtime` on Windows, breaking the "corrupt file ⇒
  ValueError" contract that `test_ptu_security.py` enforces. Wrap and re-raise as
  `ValueError`; add a security test.
- **Tag payload bound is "rest of file"** (`ptu.py:120-126`): correct, but a corrupt tag in a
  6 GB file can still demand a ~6 GB allocation. Add an absolute header-tag cap
  (e.g. 16 MiB) and a test.
- **Photons beyond the 4096-bin cap are dropped silently** (`ptu.py:167-170,197-198`): for
  `rep_time/resolution > 4096` the decay tail is discarded without a word, biasing lifetime
  fits. Warn once with the drop count; document the cap in `docs/ptu.md`.

### 2.2 `Zurich` parsing robustness — [bug/robustness]
- **No encoding on `read_text()`** (`zurich.py:45`): locale-dependent (cp1252 vs UTF-8),
  behaviour changes again with PEP 686. Use `encoding="utf-8", errors="replace"` (PTU already
  does explicit UTF-8 + latin-1 fallback). `tests/test_zurich.py:8-11` mirrors the bug —
  write fixtures with `encoding="utf-8"` and add a `µ`-bearing stream name.
- **Non-numeric two-column row crashes** (`zurich.py:70-75`): `"1.0;abc"` raises deep
  `ValueError` while a three-column row is politely skipped. Wrap the float conversions in
  `try/except` and count as skipped.
- **Stream-name heuristic is fragile** (`zurich.py:59-61`): any future comment line with a
  comma becomes a phantom stream and aborts the load with a misleading divisibility error.
  Match the exporter's actual header shape; include detected stream names in the error.
- **`average()` on 1-sample / zero-`dt` streams** (`zurich.py:118-119`): `IndexError` /
  `OverflowError`. Guard with a clear message; consider `np.median(np.diff(t))` for jitter.
- **`average()` meta reports requested, not achieved, bintime** (`zurich.py:119-123`): the
  real bin width is `bin_size * dt` (up to 50 % off). Store both; warn on large deviation.
  This matters because `Match` aligns against a PL trace binned at exactly `bintime`.
- **Reference remapping can emit duplicate trigger points** (`zurich.py:235-236`): several
  flanks can collapse onto one coarse-axis index; de-duplicate and warn.

### 2.3 `utils` edge cases — [bug]
- `feature_scale_params` / `zscore_params` crash on empty arrays (`utils.py:69,98`) despite
  docstrings promising NaN-free robustness. Return `(0.0, 0.0)` for `y.size == 0`.
- `left_bound_fraction` / `inflection_left_bound` assume a positive maximum
  (`utils.py:195-196,205`); all-negative input raises `IndexError`. Enforce/document the
  precondition, or use fraction-of-range.
- `inflection_left_bound` treats non-adjacent samples as neighbours (the `>= 10 %` index set
  can have holes) and counts `np.sign(...) == 0` as two crossings (`utils.py:206-215`).
  Restrict to the largest contiguous run and use `d2y[i-1]*d2y[i] < 0`.
- `bin_average` with a float `bin_size` fails opaquely (`utils.py:178-183`); coerce with
  `int(...)` and validate `>= 1`.
- `smooth` edge modes differ per method (`nearest` vs scipy's default `reflect`,
  `utils.py:132,136`) — matters exactly at rising edges. Unify (pass `mode="nearest"`).
- `derivative` silently yields inf on duplicated timestamps (`utils.py:147`); raise instead.

### 2.4 `Match` robustness — [bug]
- Triggers outside the shared grid are silently clamped to endpoints
  (`match.py:280-285`), so `_window_mask` bounds an unrelated region and statistics are
  quietly computed over the wrong span. `PTU._map_triggers` refuses to clamp in the same
  situation — mirror that (warn or raise).
- `scaling="bogus"` + float trigger reports the wrong error (window-mask check runs before
  the unknown-method branch, `match.py:348-364`). Validate `scaling` in `__init__` next to
  the other argument checks.

### 2.5 `PTU.fit_lifetime` / `time_trace` window validation — [bug]
- Empty fit window (`mode="manual"`, `start` beyond range) → `IndexError` at
  `ptu.py:623`; validate `t_fit.size` with a clear message (like `flid` already does at
  `ptu.py:487-488`).
- `time_trace(tstart > tend)` silently returns an empty Signal (`ptu.py:256-265`);
  validate ordering and finiteness like `bintime` already is.

### 2.6 `fitting.MultiExpFit` input validation — [bug]
`fitting.py:140-147`: 1-sample input → `IndexError` in `_default_p0`; duplicated first
sample → `log10(0)`; descending axis → NaN, surfacing as an opaque scipy error. Validate
(finite, strictly increasing, length ≥ 2·n_exp+1) up front, in the style of
`match._validate_signal`. Also: all-zero `y` silently disables the `constrain_sum_A`
penalty (`fitting.py:70`) — floor the penalty weight.

### 2.7 `selector` event-state bugs — [bug]
- `DraggableCircle`: `on_press` ignores `event.button`, and right-click delete leaves stale
  drag state → a spurious circle appears at a nonsense position on drag
  (`selector.py:344-351,366-386,414-421`). Handle button filtering in `on_press`; avoid two
  callbacks on the same event with order-dependent state.
- `InteractiveBoxSelector._on_release` returns early for non-left buttons *before* resetting
  `_mode`/`_drag_start` (`selector.py:240-242`) — the box keeps resizing with no button held.
- `_handle_size_data` derives the hit radius from x-limits only (`selector.py:82-86`);
  vertical handles are unclickable on axes with very different x/y ranges.

### 2.8 `JPK` loader robustness — [bug/robustness]
- Metadata tag split without coercion (`jpk.py:94,153`): a `bytes` private tag raises
  `TypeError`. Coerce/decode defensively like the PTU string handling.
- Multiple unnamed pages: last one silently wins for thumbnail + dx/dy
  (`jpk.py:90-95`). Accept the first, warn on others.
- Physical axes are pixel *left corners*, so `Image.extent` spans one pixel less than the
  scan size and `imshow` shifts by half a pixel (`jpk.py:141-142` + `core.py:191-201`).
  Use pixel centres `(np.arange(n) + 0.5) * d / n` and/or make `extent` edge-aware; state
  the convention in `docs/jpk.md`.

---

## 3. Cohesion / consistency (the "same datatypes everywhere" audit)

### 3.1 ROI y-filter conventions differ between methods — [consistency]
`ptu.flid` uses `(trace >= ymn) & (trace < ymx)` (`ptu.py:464`); `get_roi_matrix` uses
`<=` (`ptu.py:380`). The same `SingleBoxSelector` box selects different bins depending on
the consumer. Pick one convention and factor a shared private ROI-mask helper.

### 3.2 Trigger-producing APIs return three different shapes — [consistency]
- `Zurich.laser_triggers` → `TriggerCollection` (with `"all"`),
- `PTU.laser_triggers` → `TriggerPair` (first two markers only; extra markers ignored),
- `Zurich.find_peaks` → raw `(N, 2)` float ndarray (index coerced to float!), untested, and
  shadowing `scipy.signal.find_peaks` while returning *onsets*.
**Fix:** return `TriggerPoint` lists / `TriggerCollection` everywhere; consider renaming
`find_peaks` → `trigger_onsets`; consider letting PTU also return an `"all"` list for files
with more than two markers.

### 3.3 `fit_gaussian2d` returns a bare 4-tuple — [consistency]
`fitting.py:223-259` violates the "return `Signal`/`Image`" contract; the fitted surface is
exactly what `Image` is for, and unnamed `popt[3]`/`popt[4]` invite σx/σy swaps. Return an
`Image` with `meta={"popt", "pcov", "perr", "names"}` (or at least a named dict).

### 3.4 `print()` vs `warnings.warn` vs (missing) logging — [consistency]
Happy-path `print` calls are scattered through `ptu.py` (70, 205, 517, 661), `zurich.py`
(47, 238), `io.py` (61, 78, 83, 107), `plotting.py` (186, 254), while abnormal cases
correctly use `warnings.warn`. A library printing to stdout is noisy in notebooks and
untestable. Route informational messages through `logging.getLogger(__name__)` or a
`verbose` flag; update docs/examples together.

### 3.5 Loader label/meta conventions differ — [consistency]
PTU: `label="{stem} time trace"`, no file key in meta. Zurich: `label="{stem} {stream}"`,
`meta={"stream"}`. JPK: `label=channel` (no stem), `meta={"file": stem}`. Legend entries
from multiple JPK files are indistinguishable; only JPK records provenance.
**Fix:** one rule everywhere, e.g. `label = f"{stem} {descriptor}"` **and**
`meta["file"] = stem` in all three loaders.

### 3.6 Parameter naming collisions with the `units` mapping — [consistency]
`Zurich.signal(units: str = "V")` / `average(units=...)` (`zurich.py:104,110`) use `units`
for a y-unit *string* while `Signal.units` is the validated `{"x","y"}` mapping. Rename the
parameter (`y_unit`). Similarly `JPK.units` is a `{channel: z_unit}` dict (`jpk.py:75`) —
rename to `z_units`.

### 3.7 `utils.downsample_sum(image, fx, fy)` applies `fx` to rows — [consistency]
`utils.py:296-307`: `fx` is the y/row factor, inverted vs. the package-wide "x = columns"
convention and vs. what the docstring implies. Rename the parameters
(`factor_rows`/`factor_cols` or swap) and update the test that pins the inverted behaviour.

### 3.8 `Match` polish — [consistency]
- `_SCALING_INFO["trigger"]` labels are dead (immediately overwritten, `match.py:114-117`
  vs `248-257`); meanwhile `feature`/`zscore` *lose* the source filename in the label. Unify
  (e.g. `f"{label} (aligned, normalized)"` for all modes).
- Both x-units are assumed equal (`match.py:239`); compare `kpfm.units["x"]` and
  `timetrace.units["x"]`, warn/raise on mismatch.
- `kpfm_derivative` meta lacks `scaling`/`smoothing`/`source` and hard-codes the label
  `$d\Phi/dt$` even when z-scored (`match.py:412-419`).
- `pearson`/`cross_correlation` return plain dicts; the correlogram is a plottable curve
  and would be idiomatic as a `Signal` with `meta={"best_lag"}`. At minimum document the
  exception in `docs/match.md`.
- `_validate_signal` returns `(x, y)` but call sites discard `y` and re-derive it.
- `np.correlate(mode="full")` is O(n²); `scipy.signal.correlate(..., method="auto")` is a
  drop-in FFT-backed replacement (the auto grid can be tens of thousands of points).
- Grid sizing `int(max(span/res, 2)) + 1` truncates and floors at 3 where docs say 2
  (`match.py:264-267`); use `round` and `max(..., 2)`.

### 3.9 `PTU` API polish — [consistency]
- `flid` returns `Image | list[Image]` depending on the `isolated` flag buried among 18
  parameters (`ptu.py:385-392`) — the only shape-varying return in the package. Always
  return a list, or split into `flid()` / `flid_components()`. Consider grouping the 8
  window/ROI keywords into the ROI dict. Also: all returned Images share the same *mutable*
  `base_meta` dict (`ptu.py:521-541`) — copy per image.
- `_tt_cache` is silently repointed by *any* `time_trace` call, including internal windowed
  ones from `flid`/`get_roi_matrix` (`ptu.py:305-315`), so a subsequent `threshold()` or
  `lifetime_histogram()` quietly analyses a window instead of the full file. Only cache
  full-range calls, or use a cache-bypassing private path internally.
- Repeated positive-int validation boilerplate (`ptu.py:437-446`) → extract
  `_positive_int(...)`; `taus`/`counts` are rebound from `list` to `ndarray` mid-function.
- Per-record Python loop (`ptu.py:177-204`) is ~100× slower than a vectorised
  `np.frombuffer` + bit-mask + `cumsum` decode; worth it for multi-minute files.

### 3.10 `core.Signal`/`Image` container polish — [consistency]
- `core._as_mapping` (`core.py:247-251`) is dead code — delete.
- `meta` is stored by reference while `units` is defensively copied
  (`core.py:67,79`) — copy `meta` too.
- `len(Signal)` = number of samples, but `keys()` has 5 entries and `dict(signal)` has
  len 5; `Image` has no `__len__` at all. Prefer an explicit `n_samples`/`size` property;
  at minimum document the discrepancy and align `Image`.
- `extent` is centre-based while `imshow` expects pixel edges (half-pixel error;
  see 2.8), and descending axes are silently reoriented by `min`/`max`
  (`core.py:191-201`). Decide a convention, document it, or return
  `(x[0], x[-1], y[0], y[-1])`.
- `extent_scaled`'s `KeyError` message blames the caller's `unit` even when a *stored*
  unit is invalid (`core.py:216-222`); split the two error messages.
- All JPK `Image`s share the same axis arrays by reference (`jpk.py:243-244`); an in-place
  `img.x *= 1e6` corrupts every sibling image. Pass copies.

### 3.11 `database` polish — [consistency]
- Absolute Windows paths in JSON make the databank machine-bound; `load_json` never checks
  existence, and `Entry.load()` surfaces a bare `FileNotFoundError`. Persist a versioned
  envelope `{"version", "root", "entries"}` with root-relative paths; add `Entry.exists()`
  and a "rebuild with build_database(root)" error message.
- `_TIME_RE` requires `[_.]` *after* the `s` but matches against `path.stem` (extension
  stripped), so `..._360s.ptu` silently falls back to the 360 s default
  (`database.py:75`). Use `(?=[_.]|$)`; record which fields were defaulted so parsed and
  guessed values are distinguishable.
- `or`-chained fallbacks treat legitimate `0.0` as missing (`database.py:341-343`);
  line 344 already uses `is not None` — make all consistent.
- `filter_entries` with a typo'd field raises `AttributeError`; `add_entry(**overrides)`
  raises a raw `TypeError` for unknown keys **and silently discards** `txt_path`/`ptu_path`
  overrides (`database.py:423,469`). Validate against `dataclasses.fields(Entry)`.
- `save_csv([])` writes a zero-byte file; derive `fieldnames` from `fields(Entry)` and
  always write the header. Note "export only" (no `load_csv`) in the docstring.
- `_pair_leaf_files` docstring says unmatched files are dropped "with a `print` warning";
  the code (correctly) warns — update the docstring and mention the warnings in
  `docs/database.md`.

### 3.12 `io.sort_measurement_files` silently drops duplicates — [bug/consistency]
`io.py:93-99` keys on suffix per (wavelength, power) group, so `_000.ptu` and `_001.ptu`
with identical tokens overwrite each other with no warning. Store lists per suffix or warn
on overwrite.

### 3.13 `Zurich.laser_triggers` threshold is hard-coded — [consistency]
`zurich.py:163` uses a magic `0.1` while the sibling slope path exposes
`slope_threshold: float = 0.1`. Add `threshold: float = 0.1` and document both.

### 3.14 `JPK.get_save_path` — [style]
`jpk.py:274-278`: only public method without a docstring or return annotation, trailing
whitespace, and a `get_*` name with a filesystem side effect (`mkdir` — also missing
`parents=True`). Document, annotate `-> Path`, use `mkdir(parents=True, exist_ok=True)`,
and consider a `create: bool` parameter or renaming.

---

## 4. Compatibility, packaging, tests

### 4.1 `tests/test_public_api.py` hard-fails without matplotlib — [compat]
It imports `plotting`/`selector` unconditionally (`test_public_api.py:7-27`), so the suite
errors on a core install even though matplotlib is optional and `test_plotting.py`
correctly uses `importorskip`. Split into core + matplotlib sets (skip the latter).

### 4.2 The headless-import contract has no test — [compat]
Nothing asserts that `import kpfm_pl` avoids matplotlib/tkinter — the single most
load-bearing invariant in AGENTS.md. Add a subprocess test
(`assert 'matplotlib' not in sys.modules and 'tkinter' not in sys.modules`). Also assert
`_PLOTTING_NAMES == plotting.__all__` and `_SELECTOR_NAMES == selector.__all__`
(`__init__.py:121-135` is maintained by hand and will drift).

### 4.3 `pyproject.toml` packaging — [compat]
`license = { text = "GPL-3.0" }` is deprecated under PEP 639 (setuptools ≥ 77 warns,
removal announced) → `license = "GPL-3.0-only"` + `license-files = ["LICENSE"]`. No
`classifiers` (Python 3.10/3.11/3.12/3.13, license). `py.typed` is correctly declared.

### 4.4 `tests/test_public_api.py` hard-codes the version — [style]
Asserting `__version__ == "1.3.0"` means every release edits a test. Assert single-source
identity + a `\d+\.\d+\.\d+` format instead.

### 4.5 `kpfm_pl.plotting` unreachable via attribute access — [style]
After `import kpfm_pl`, `kpfm_pl.plotting` raises `AttributeError` (only
`from kpfm_pl import plotting` works). Two lines in the existing `__getattr__` fix it.
Also `__dir__` surfaces privates (`_PLOTTING_NAMES`, `annotations`, ...).

### 4.6 `AGENTS.md` layout drift — [compat/docs]
AGENTS.md says examples live in `src/examples/` with the test data; they are at top-level
`examples/` (+ `examples/Test data/`). README and `docs/match.md` are correct; AGENTS.md is
the file agents read first — fix its three references. Related: `tests/test_ptu.py` /
`test_jpk.py` skip silently if `examples/Test data/` moves — resolve the data dir in a
`conftest.py` and make accidental relocation loud.

### 4.7 Examples (living documentation) — [consistency]
- Both correlate examples build `str(...)` paths purely so `file.endswith('.ptu')` works —
  the residue of the recent `WindowsPath.endswith` crash, i.e. the workaround, not the fix.
  Keep `Path` objects and dispatch on `p.suffix.lower() == ".ptu"` (the loaders already
  accept `str | Path`).
- `import kpfm_pl as k` then `k = m.kpfm_signal()` rebinds the package alias
  (`example_correlate*.py`); rename to `k_sig`/`t_sig` as `docs/match.md` already does.
- No example guards the documented `None` returns (`laser_triggers`, `get_markers`,
  `make_circle`) — each is a latent `TypeError`.
- Style drift: mixed import idioms, mixed `afm.data` / `afm["data"]`, an unused
  `numpy` import in `example_ptu.py`, and `example_ptu.py:37` re-reading a marker-based
  file with `laser_pulse=False` (the legacy path) in the *current* example.

### 4.8 Documentation drift (summary) — [docs]
- `docs/zurich.md`: stale stream-name table (1.1), presents `KPFM_STREAM`/`TRIGGER_STREAM`
  as importable although they're in no `__all__`; hard-coded threshold undocumented;
  `None`-return paths incomplete.
- `docs/ptu.md`: `get_roi_matrix` description doesn't match the code (1.2);
  `meta` keys `bin_durations`/`acquisition_duration` missing; CW-file attribute caveat
  missing; 4096-bin cap undocumented; `dark_counts=2.25` vs `2.55` in the package
  docstring example.
- `docs/jpk.md`: "unnamed 64x64 page" hard-codes a size the loader never assumes;
  pixel-convention caveat missing.
- `plotting` docstring "Contents" omits `crop_roi` (public + exported, documented nowhere);
  `selector` docstring omits `SingleBoxSelector` and describes a `Delete` binding while
  `crop_roi`'s UI says right-click.
- No `docs/core.md` for the `Signal`/`Image` contract that every other manual links into.

### 4.9 Error-message quality — [style]
Unknown stream/channel lookups raise bare `KeyError` with no candidates
(`zurich.py:106,117,158`; `jpk.py:235`) — the single most likely user error given names
vary per export. Add lookup helpers that list the available names.

### 4.10 Minor plotting/fitting nits — [style]
- `plotting.py:179` commented-out leftover; `set_cbar_formatter` yields NaN format choice on
  all-NaN data (`plotting.py:111-113`); `crop_roi` doesn't reject inverted/degenerate ROIs
  (`plotting.py:259-263`) and its docstring should state it takes a raw array, not an
  `Image`.
- `fitting.legend_label` hard-codes `\mathrm{ns}` (add a `unit` parameter);
  `peak_power_density` lacks a return annotation and a zero guard.
- `tests/test_database.py:139-146` uses `try/except assert False` instead of
  `pytest.raises`.

---

## Compatibility checks that passed (no action)

- **Python ≥ 3.10 claim holds** — `from __future__ import annotations` everywhere; the only
  runtime 3.10+ feature is `zip(strict=True)`.
- **NumPy 2 ready** — no removed aliases; only `np.polyfit/np.polyval` are soft-legacy
  (see 1.9 fix).
- **SciPy ≥ 1.9 APIs** all current; scipy lazily imported where optional.
- **Headless contract holds in code** — no top-level matplotlib outside `plotting.py`
  (which is itself lazy); tkinter lazy in `io.py`. (Just untested — see 4.2.)
- **Loaders accept `str | Path` consistently** — the recent `WindowsPath.endswith` bug was
  in the example script only; no `.endswith` on paths survives in `src/`.
- **PTU length-prefix bounds check** is present and correct (keep it; see hardening 2.1).

---

## Suggested fix order

1. **1.1–1.6** — user-visible wrong results with current data (stream names, ROI matrix,
   lag sign, float-trigger scaling, flat-root database crash, mirrored images).
2. **1.7–1.15** — silent data corruption / wrong numbers (time flip, uint y, set_zero NaN,
   threshold truncation, matrix IndexError, CW objects, mis-pairing, weighted r², images([])).
3. **Section 2** — robustness of the parsers (PTU untrusted-input items first) and edge
   cases, each landed with its test.
4. **Section 3** — API cohesion; do 3.2 (trigger shapes) and 3.4 (print→logging) first, as
   they touch public behaviour and docs.
5. **Section 4** — packaging, test-suite gaps, examples, and the docs-drift table.
