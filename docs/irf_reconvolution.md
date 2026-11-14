# IRF reconvolution – how it works

> **Status:** background explanation. IRF reconvolution is **implemented
> since v1.7.0** (`PTU.fit_lifetime(irf=...)`, GUI *Lifetime fit → use IRF*);
> without an IRF the lifetime fit is a *tail fit*. Section 8 describes the
> implementation; usage and parameters are in [ptu.md](ptu.md#irf-reconvolution)
> and [gui.md](gui.md). Bracketed numbers refer to the reference list at the
> end.

## 1. Why the IRF matters

A TCSPC lifetime histogram is not the true decay of the sample. Every photon
arrival time is blurred by the *instrument response function* (IRF) `R(t)`:
laser pulse width, timing jitter of the electronics and, usually dominant,
the transit-time spread of the detector. The measured histogram is the
**convolution** of the true decay `F(t)` with the IRF, plus a background
`B` [1–5]:

```
I(t) = ∫ R(t') F(t − t') dt' + B ,      F(t) = Σᵢ aᵢ exp(−t/τᵢ)   (t ≥ 0)
```

There are two ways to deal with this:

- **Tail fit** (what `kpfm_pl` does today). Fit `Σ aᵢ exp(−t/τᵢ)` only to the
  part after the maximum, where the IRF has (nearly) died away. This is fine
  when all lifetimes are much longer than the IRF width; PicoQuant
  recommends it only for "lifetimes … significantly longer than the FWHM of
  the instrument response function" and notes that the start of the fit range
  "is usually a bit arbitrary" [7]. Because `t = 0` is put at the peak (or
  the fit start), the amplitudes depend on that choice: each `aᵢ` is scaled by
  `exp(t_start/τᵢ)`, which biases amplitude ratios and intensity fractions
  towards the long components. The lifetimes themselves are robust only if
  `τ ≫ FWHM`.
- **Reconvolution fit** (iterative reconvolution). Convolve the model decay
  with the *measured* IRF and fit the **whole** histogram, including the rising
  edge [1, 9, 10]. Short components that are hidden under the IRF become
  accessible, and the amplitudes refer to the true excitation time, so they no
  longer depend on an arbitrary `t = 0`. Edinburgh Instruments quote about
  1/10 of the IRF FWHM as the shortest lifetime resolvable by reconvolution
  under ideal conditions and about 1/3 FWHM as a more practical limit [8].

Rule of thumb (not from a single authoritative source): use reconvolution
whenever a lifetime is within roughly 5–10× the IRF FWHM, or whenever you want
to interpret amplitudes or intensity fractions.

## 2. The model that is fitted

On a grid of lifetime bins `tⱼ` (width `Δ`), the model used by commercial and
open-source packages [5, 7, 24, 25] is

```
m(tⱼ) = Σᵢ aᵢ · [ R(t − δ) ⊗ exp(−t/τᵢ) ](tⱼ)  +  s · R(tⱼ − δ)  +  B
```

| Parameter | Meaning |
|---|---|
| `aᵢ`, `τᵢ` | amplitudes and lifetimes of the true (IRF-free) decay |
| `δ` | **IRF shift**: time offset between the recorded IRF and the decay |
| `B` | constant background (dark counts, ambient light, after-pulsing) |
| `s` | optional **scatter** contribution: laser light leaking into the detection path, which has the shape of the IRF itself |

The IRF `R` is the histogram of the IRF file after **background subtraction**
(a constant: the mean of the flat floor far from the peak) and
**normalisation to unit area** (`Σ R = 1`) [5, 24]. Without the background
subtraction a flat IRF floor, convolved with the decay, produces a spurious
constant or slow component. With the unit-area normalisation the `aᵢ` carry
the photon counts.

The subtracted floor must **not be clipped at zero**, even though clipping
negative values is a common recipe: it removes only the negative half of the
floor's Poisson noise and leaves a positive floor of about `0.4·√floor`
counts per bin. The convolution turns it into a flat level that the fit
takes from the background `B`. On a synthetic test file this cost 4 % of
`B`; with a noisier IRF floor it took 90 % of it (fixed in v1.7.2). Likewise
the floor level is estimated as a mean, not a median, because the median of
Poisson counts is biased.

### IRF shift `δ`

The IRF and the decay never line up perfectly: cable/sync drift between the
two recordings, count-rate dependent timing (≈ 20 ps observed with PMTs [4])
and above all the **colour effect** – many detectors, especially SPADs, have
a wavelength-dependent timing response, so an IRF recorded at the laser
wavelength is slightly shifted against fluorescence at a longer wavelength
[7]. Therefore `δ` is fitted, with sub-bin precision (cubic interpolation in
FLIMfit [24], Fourier shift or linear interpolation elsewhere [27]). A shift
cannot correct a change of the IRF *shape* (e.g. the diffusion tail of a SPAD
that depends on wavelength). `δ` and very short lifetimes are correlated, so
the shift needs a good starting value (e.g. from cross-correlating the IRF
with the decay) and sensible bounds.

### Periodic excitation (incomplete decay)

The laser fires every `T` (100 ns at 10 MHz). If a decay has not died away
within one period, the tail of the previous pulses still contributes to the
current window. SPCImage calls the correction the "incomplete model" [5],
FLIMfit sums pulses at `t = −nT` [24], and tttrlib uses a periodic
convolution [25]. For exponentials the infinite sum has a closed form:

```
Σₙ exp(−(t + nT)/τ) = exp(−t/τ) / (1 − exp(−T/τ)),     0 ≤ t < T
```

so the periodic decay is the single-pulse decay times a constant factor and
the convolution with the IRF becomes *circular* over one period. At
`T = 100 ns` this is a 0.7 % effect for `τ = 20 ns` and 16 % for `τ = 50 ns`.
A side effect matters in practice: with an incomplete decay the bins before
the rising edge contain fluorescence from the previous pulse, so the
background `B` can no longer be read off those bins [5]. The reconvolution
fit covers them because it fits the whole period; the tail fit can include
them with `prerise=True` (since v1.8.0, see
[ptu.md](ptu.md#background-and-the-bins-before-the-rise)).

## 3. Fitting statistics

TCSPC bin counts are Poisson distributed. Three objective functions are in use:

| Objective | Formula | Behaviour |
|---|---|---|
| Neyman χ² (`σ² = data`) | `Σ (yⱼ − mⱼ)² / max(yⱼ, 1)` | classic TCSPC least squares; `kpfm_pl`'s current *Poisson* weighting. Biased at low counts (bins that fluctuate low get more weight). |
| Pearson χ² (`σ² = model`) | `Σ (yⱼ − mⱼ)² / mⱼ` | biased the other way when minimised directly |
| Poisson maximum likelihood (deviance) | `2 Σ [mⱼ − yⱼ + yⱼ ln(yⱼ/mⱼ)]` | unbiased estimator; also a goodness-of-fit measure (≈ χ²-distributed) [17] |

Least squares underestimates lifetimes by about 5 % at 1 000–20 000 photons,
whereas the maximum-likelihood estimator (MLE) stays accurate [15]. Similar
conclusions are reached in [12, 16, 18, 19]. For well-filled histograms
(many counts in every fitted bin) the methods agree. For per-bin fits
(lifetime trace, FLID) with few photons, MLE is clearly preferable. Fits must
use **raw counts**: normalising the histogram to its maximum destroys the
Poisson statistics.

A good fit has a reduced χ² (or deviance per degree of freedom) close to 1
and residuals that scatter randomly around zero, *including on the rising
edge*. The degrees of freedom are `n_bins − k`, where `k` counts **all** free
parameters, including `δ`, `B` and `s`.

## 4. Choosing the number of exponentials: BIC

Adding an exponential always lowers χ², even if the extra component only fits
noise. The **Bayesian information criterion** [21, 22] penalises extra
parameters:

```
BIC = −2 ln L̂ + k ln n
```

with `L̂` the maximised likelihood, `k` the number of free parameters and `n`
the number of data points (here: fitted histogram bins). For a χ² fit with
known Poisson errors `−2 ln L = χ² + const`, so `BIC = χ² + k ln n`; for the
Poisson likelihood `BIC = deviance + k ln n + const`. Constants cancel when
two models are compared **on the same data, fit window and statistic** – BIC
values from different histograms or windows are not comparable.

`ΔBIC = BIC(n exp) − BIC(n+1 exp)` approximates twice the log Bayes factor in
favour of the extra component. Kass & Raftery [22] grade the evidence:

| ΔBIC | Evidence for the more complex model |
|---|---|
| 0 – 2 | not worth more than a bare mention |
| 2 – 6 | positive |
| 6 – 10 | strong |
| > 10 | very strong |

Caveats: BIC assumes the model family is right. With very many counts, any
imperfection of the IRF (shape, colour effect) makes BIC favour ever more
components. Reject an extra component anyway if amplitudes collapse to zero,
two lifetimes merge or a lifetime runs into a bound.

## 5. Recording a good IRF

- Same detector, filters, TCSPC resolution (bin width), sync rate/divider,
  sync offset/cable delays and CFD settings as the sample measurement [7].
- Low count rate (well below the sample's, and far below the sync rate) to
  avoid pile-up [7]; dilute the scatterer rather than using ND filters.
- Many counts: IRF noise propagates directly into the model.
- Scattered laser light is the usual IRF source. For SPAD detectors with a
  strong colour effect, a quenched dye (e.g. KI-quenched fluorescein or
  erythrosine B) emitting near the sample's wavelength, or SHG for two-photon
  excitation, gives a more faithful IRF [7]. Alternatively a reference dye of
  known lifetime replaces the IRF (delta-function convolution method [11]).

## 6. Interpreting the results

After reconvolution the `aᵢ` are the amplitudes of the IRF-free decay at the
excitation time. Derived quantities:

- intensity fractions `fᵢ = aᵢτᵢ / Σ aⱼτⱼ`
- amplitude-weighted lifetime `τ_amp = Σ aᵢτᵢ / Σ aᵢ`
- intensity-weighted lifetime `τ_int = Σ aᵢτᵢ² / Σ aᵢτᵢ` (as reported by
  `kpfm_pl` today [5])

Unlike in a tail fit, none of these depends on where `t = 0` is put.

## 7. Pitfalls specific to `kpfm_pl` data

- **Settings mismatch.** The IRF file and the sample file must have the same
  `MeasDesc_Resolution` and repetition period; a mismatch is reported as an
  error.
- **Lifetime window shorter than a period.** `kpfm_pl` caps the lifetime axis
  at 4096 bins. Choose the TCSPC resolution so that the window covers a whole
  laser period (e.g. ≥ 25 ps at 10 MHz); the reconvolution fit refuses files
  where it does not, because the periodic (circular) convolution needs the
  full period.
- **Circular lifetime shift** (`lifetime_shift`): handled. The IRF shift `δ`
  is applied circularly over the period, so a sample whose lifetime axis was
  rotated still matches an unrotated IRF; `δ` then also contains that offset.
- **Partial last bin.** If the period is not a whole number of bins, its
  last bin is only partly covered and is left out of the fit.
- **Correlations.** `δ` ↔ short `τ`, `s` ↔ very short `τ`, `B` ↔ `τ ≳ T`.
  Bound parameters, fix what is known, and only free `s` if scatter is
  expected.
- **Fit window.** A reconvolution fit must include the rising edge and some
  bins before it. The automatic fit-start modes therefore fit the whole
  period; *Manual* fits only the bins between start and end.

## 8. Implementation in `kpfm_pl`

Already available (v1.5.0): the Poisson MLE (`weighting="mle"`), the BIC of
every lifetime fit with the ΔBIC against the previous fit in the GUI, the
over-fitting warnings, and the incomplete-decay correction of the tail fit
(`A = A_fit·(1 − exp(−T/τ))`, section 2). Since v1.8.0 the tail fit can
also fit its background (`fit_background=True`) and the bins before the
rise (`prerise=True`), and every fit with a fitted background warns about
lifetimes above half the laser period (the `B` ↔ `τ ≳ T` correlation of
section 7).

Implemented in v1.7.0 (`fitting.IRFReconvolutionFit`, `PTU.fit_lifetime(irf=...)`;
verified on synthetic data: kernel against brute-force integration, unbiased
recovery of τ, δ, B and s):

- **Model:** `mⱼ = Σᵢ aᵢ [g_δ ⊛ k_τᵢ]ⱼ + s·g_δ,ⱼ + B` on one laser period,
  with `g` the background-subtracted, unit-area IRF, `g_δ` its sub-bin shifted
  copy (Fourier shift), `k_τ` the bin-integrated periodic exponential kernel
  and `⊛` a circular FFT convolution. The periodic kernel contains the
  incomplete-decay correction, so `aᵢ` are per-pulse amplitudes. The
  *Incomplete decay* toggle (v1.6.0) switches to a single-pulse kernel.
- **Free parameters:** `aᵢ ≥ 0`, `τᵢ`, `δ` (started where the half-maximum
  rising edges of IRF and decay coincide, bounded to ± max(20 bins, 5 IRF
  FWHM), at most a quarter period); optional checkboxes to fit `B`
  (otherwise fixed from the dark counts) and the scatter `s`. `k` in the BIC
  counts all of them.
- **IRF preparation:** the floor is subtracted (mean of the floor bins) but
  not clipped, see section 2.
- **Statistic:** any `weighting` (least squares, Poisson, Poisson MLE).
- **Lifetime trace / FLID:** the per-bin fits reconvolve with `δ` (and `B`,
  `s`) taken from the last *Lifetime fit*; without one they fall back to the
  tail fit and log a warning.
- **GUI:** one IRF `.ptu` file for the whole application, remembered across
  sessions and changeable at any time, a *Use IRF (reconvolution)* checkbox,
  and the fitted shift `δ` in the results table.

## References

1. J. R. Lakowicz, *Principles of Fluorescence Spectroscopy*, 3rd ed., Springer, 2006, ch. 4 "Time-Domain Lifetime Measurements", pp. 97–155. doi:10.1007/978-0-387-46312-4
2. D. V. O'Connor, D. Phillips, *Time-Correlated Single Photon Counting*, Academic Press, 1984. ISBN 978-0-12-524140-3
3. W. Becker, *Advanced Time-Correlated Single Photon Counting Techniques*, Springer Series in Chemical Physics 81, Springer, 2005. doi:10.1007/3-540-28882-1
4. W. Becker, *The bh TCSPC Handbook*, Becker & Hickl GmbH, Berlin (free download, several editions). https://www.becker-hickl.com/literature/documents/flim/the-bh-tcspc-handbook/
5. Becker & Hickl, *SPCImage 2.9 – Data Analysis Software for FLIM*, manual, 2007 (model, scatter, offset, "incomplete model", τ_mean).
6. M. Wahl, *Time-Correlated Single Photon Counting*, PicoQuant Technical Note.
7. PicoQuant / tcspc.com knowledge base: "How to measure the instrument response function (IRF)", "Lifetime fitting using the TCSPC fitting script", "Reconvolution fit". https://www.tcspc.com
8. Edinburgh Instruments, Technical Note TN P81, *Instrument Response Function Widths & Minimum Measurable Lifetimes in the FLS1000*, 2023.
9. A. Grinvald, I. Z. Steinberg, "On the analysis of fluorescence decay kinetics by the method of least-squares", *Anal. Biochem.* **59**, 583 (1974). doi:10.1016/0003-2697(74)90312-1
10. D. V. O'Connor, W. R. Ware, J. C. André, "Deconvolution of fluorescence decay curves. A critical comparison of techniques", *J. Phys. Chem.* **83**, 1333–1343 (1979). doi:10.1021/j100473a019
11. M. Zuker, A. G. Szabo, L. Bramall, D. T. Krajcarski, B. Selinger, "Delta function convolution method (DFCM) for fluorescence decay experiments", *Rev. Sci. Instrum.* **56**, 14–22 (1985). doi:10.1063/1.1138457
12. Ž. Bajzer, T. M. Therneau, J. C. Sharp, F. G. Prendergast, "Maximum likelihood method for the analysis of time-resolved fluorescence decay curves", *Eur. Biophys. J.* **20**, 247–262 (1991). doi:10.1007/BF00450560
13. M. Köllner, J. Wolfrum, "How many photons are necessary for fluorescence-lifetime measurements?", *Chem. Phys. Lett.* **200**, 199–204 (1992). doi:10.1016/0009-2614(92)87068-Z
14. J. Enderlein, R. Erdmann, "Fast fitting of multi-exponential decay curves", *Opt. Commun.* **134**, 371–378 (1997). doi:10.1016/S0030-4018(96)00384-7
15. M. Maus, M. Cotlet, J. Hofkens, T. Gensch, F. C. De Schryver, J. Schaffer, C. A. M. Seidel, "An experimental comparison of the maximum likelihood estimation and nonlinear least-squares fluorescence lifetime analysis of single molecules", *Anal. Chem.* **73**, 2078–2086 (2001). doi:10.1021/ac000877g
16. D. A. Turton, G. D. Reid, G. S. Beddard, "Accurate analysis of fluorescence decays from single molecules in photon counting experiments", *Anal. Chem.* **75**, 4182–4187 (2003). doi:10.1021/ac034325k
17. S. Baker, R. D. Cousins, "Clarification of the use of chi-square and likelihood functions in fits to histograms", *Nucl. Instrum. Methods* **221**, 437–442 (1984). doi:10.1016/0167-5087(84)90016-4
18. T. Hauschild, M. Jentschel, "Comparison of maximum likelihood estimation and chi-square statistics applied to counting experiments", *Nucl. Instrum. Methods A* **457**, 384–401 (2001). doi:10.1016/S0168-9002(00)00756-7
19. T. A. Laurence, B. A. Chromy, "Efficient maximum likelihood estimator fitting of histograms", *Nat. Methods* **7**, 338–339 (2010). doi:10.1038/nmeth0510-338
20. H. Dembinski, M. Schmelling, R. Waldi, "Application of the iterated weighted least-squares fit to counting experiments", arXiv:1807.07911.
21. G. Schwarz, "Estimating the dimension of a model", *Ann. Statist.* **6**, 461–464 (1978). doi:10.1214/aos/1176344136
22. R. E. Kass, A. E. Raftery, "Bayes factors", *J. Am. Stat. Assoc.* **90**, 773–795 (1995). doi:10.1080/01621459.1995.10476572
23. P. R. Bevington, D. K. Robinson, *Data Reduction and Error Analysis for the Physical Sciences*, 3rd ed., McGraw-Hill, 2003.
24. S. C. Warren et al., "Rapid global fitting of large fluorescence lifetime imaging microscopy datasets", *PLoS ONE* **8**, e70687 (2013). doi:10.1371/journal.pone.0070687
25. T.-O. Peulen et al., "tttrlib: modular software for integrating fluorescence spectroscopy, imaging, and molecular modeling", *Bioinformatics* **41**, btaf025 (2025). doi:10.1093/bioinformatics/btaf025
26. D. Gottlieb et al., "FLUTE: a Python GUI for interactive phasor analysis of FLIM data", *Biological Imaging* **3**, e21 (2023). doi:10.1017/S2633903X23000211
27. LifeFit (Python TCSPC analysis package), https://lifefit.readthedocs.io
