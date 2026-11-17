"""Second-order correlation (g2, antibunching) of T2 files and its helpers."""

import numpy as np
import pytest
from ptu_writer import (
    GENERIC_T2,
    GENERIC_T3,
    PICOHARP_T2,
    antibunching_events,
    encode_t2,
    encode_t3,
    write_ptu,
)

from kpfm_pl import PTU, peak_delay, photon_correlation, pulsed_g2_zero

TICK = 1e-11          # 10 ps time tags
PERIOD = 10000        # 100 ns laser period, in ticks


# ------------------------------------------------------------------ helpers
def test_photon_correlation_matches_brute_force():
    rng = np.random.default_rng(1)
    a = np.sort(rng.integers(0, 100000, 400))
    b = np.sort(rng.integers(0, 100000, 500))
    lags, counts = photon_correlation(a, b, max_lag=2000, bin_width=50)
    assert lags[0] == -2000 and lags[-1] == 2000 and lags.size == 81
    diff = (b[None, :] - a[:, None]).ravel()
    k = (diff + 25) // 50
    expected = np.bincount(k[np.abs(k) <= 40] + 40, minlength=81)
    np.testing.assert_array_equal(counts, expected)
    # the result does not depend on how the pairs are split into blocks
    _, small = photon_correlation(a, b, max_lag=2000, bin_width=50, max_pairs=7)
    np.testing.assert_array_equal(small, counts)


def test_photon_correlation_sorts_reports_progress_and_validates():
    calls = []
    lags, counts = photon_correlation([30, 10, 20], [21, 11], 15, 10,
                                      progress=lambda d, t: calls.append((d, t)))
    assert calls[-1] == (3, 3)
    # pairs b - a: 11-10=1, 21-20=1, 11-20=-9, 21-10=11, 21-30=-9, 11-30=-19; the
    # range is rounded up to whole bins (-25 ... 24), so -19 is in the -20 bin
    assert dict(zip(lags.tolist(), counts.tolist())) == {-20: 1, -10: 2, 0: 2, 10: 1, 20: 0}
    with pytest.raises(ValueError, match="bin_width"):
        photon_correlation([1], [2], 10, 0)
    with pytest.raises(ValueError, match="max_lag"):
        photon_correlation([1], [2], 0, 1)


def _comb(centre_area, side_area, period=100.0, delay=12.0, n_side=4, step=1.0):
    """A correlogram of Gaussian peaks at delay + k * period."""
    lags = np.arange(-n_side * period - 50, n_side * period + 50 + step, step)
    counts = np.zeros(lags.size)
    for k in range(-n_side, n_side + 1):
        area = centre_area if k == 0 else side_area
        counts += area * step / (3 * np.sqrt(2 * np.pi)) * np.exp(
            -0.5 * ((lags - delay - k * period) / 3.0) ** 2)
    return lags, counts


def test_pulsed_g2_zero_from_peak_areas():
    lags, counts = _comb(20.0, 100.0)
    assert peak_delay(lags, counts, 100.0) == pytest.approx(12.0, abs=1.0)
    out = pulsed_g2_zero(lags, counts, 100.0)
    assert out["g2_0"] == pytest.approx(0.2, abs=1e-3)
    assert out["delay"] == pytest.approx(12.0, abs=1.0)
    assert sorted(out["side_areas"]) == [-4, -3, -2, -1, 1, 2, 3]   # k=4 window not whole
    assert out["n_side_peaks"] == 7 and out["window"] == 50.0
    narrow = pulsed_g2_zero(lags, counts, 100.0, delay=12.0, window=10.0)
    assert narrow["g2_0"] == pytest.approx(0.2, abs=1e-3)
    assert 4 in narrow["side_areas"]                                 # fits with a narrow window
    # an antibunched comb (no centre peak) is still located by its side peaks
    lags, counts = _comb(0.0, 100.0, delay=-30.0)
    assert peak_delay(lags, counts, 100.0) == pytest.approx(-30.0, abs=1.0)
    with pytest.raises(ValueError, match="side peak"):
        pulsed_g2_zero(lags[np.abs(lags) < 120], counts[np.abs(lags) < 120], 100.0)
    with pytest.raises(ValueError, match="window"):
        pulsed_g2_zero(lags, counts, 100.0, window=60.0)


# ------------------------------------------------------------------ files
def _multiharp(tmp_path, rng, name="mh.ptu", **kwargs):
    events, ticks = antibunching_events(rng, period=PERIOD, **kwargs)
    n_pulses = kwargs.get("n_pulses", 200000)
    path = write_ptu(tmp_path / name, encode_t2(events, "generic", 2),
                     record_type=GENERIC_T2, global_resolution=TICK,
                     acquisition_ms=int(np.ceil(n_pulses * PERIOD * TICK * 1e3)),
                     sync_rate=int(round(1 / (PERIOD * TICK))))
    return PTU(path), ticks


def test_single_emitter_is_antibunched_and_poisson_light_is_not(tmp_path):
    rng = np.random.default_rng(2)
    single, _ = _multiharp(tmp_path, rng, "single.ptu", single=True, brightness=0.1)
    assert single.has_lifetime and single.channels == (1, 2)
    g2 = single.g2()
    meta = g2.meta
    assert meta["method"] == "pulsed" and meta["period"] == pytest.approx(PERIOD * TICK)
    assert meta["g2_0"] < 0.1 and meta["mean_side_area"] > 200
    assert meta["delay"] == pytest.approx(0.0, abs=2e-9)           # no cable delay simulated
    assert g2.units == {"x": "s", "y": "coincidences"}
    assert g2.x[0] == pytest.approx(-5.5 * PERIOD * TICK, rel=0.01)
    assert meta["lag_bin"] == pytest.approx(PERIOD * TICK / 200)
    # normalised: side peaks of uncorrelated pulses integrate to about one period
    side = np.abs(g2.x - PERIOD * TICK) < PERIOD * TICK / 2
    assert np.sum(meta["g2"][side]) * meta["lag_bin"] / (PERIOD * TICK) == \
        pytest.approx(1.0, rel=0.1)

    poisson, _ = _multiharp(tmp_path, rng, "poisson.ptu", single=False, brightness=0.1)
    assert poisson.g2().meta["g2_0"] == pytest.approx(1.0, abs=0.15)


def test_g2_counts_every_pair_of_the_two_detectors(tmp_path):
    rng = np.random.default_rng(3)
    p, ticks = _multiharp(tmp_path, rng, single=False, brightness=0.05, n_pulses=20000)
    g2 = p.g2(max_lag=2e-7, lag_bin=1e-9)
    lags, counts = photon_correlation(ticks[1], ticks[2], 2e-7 / TICK, 100)
    np.testing.assert_array_equal(g2.y, counts)
    np.testing.assert_allclose(g2.x, lags * TICK)
    assert g2.meta["photons"] == {1: ticks[1].size, 2: ticks[2].size}
    swapped = p.g2(channels=(2, 1), max_lag=2e-7, lag_bin=1e-9)
    np.testing.assert_array_equal(swapped.y, counts[::-1])        # lag sign flips


def test_g2_of_the_on_state_and_an_roi(tmp_path):
    rng = np.random.default_rng(4)
    n_pulses = 400000                                               # 40 ms
    dark = (np.arange(n_pulses) // 50000) % 2 == 1                 # off every other 5 ms
    p, _ = _multiharp(tmp_path, rng, single=True, brightness=0.2, background=0.004,
                      n_pulses=n_pulses, dark_pulses=dark)
    trace = p.detector_trace(bintime=1e-3, channels=(1, 2))
    assert trace.y.size == 40 and trace.meta["channels"] == [1, 2]
    whole = p.g2(bintime=1e-3)
    on = p.g2(bintime=1e-3, y_min=0.5 * trace.y.max())
    assert on.meta["selected_bins"] == 20
    assert on.meta["duration"] == pytest.approx(0.02, rel=1e-6)
    # the background of the off state fills the centre peak of the whole file
    assert on.meta["g2_0"] < whole.meta["g2_0"]
    box = p.g2({"x_min": 0.0, "x_max": 0.005, "y_min": None, "y_max": None}, bintime=1e-3)
    assert box.meta["selected_bins"] == 5 and box.meta["roi"]["x_max"] == 0.005
    with pytest.raises(ValueError, match="no time bins"):
        p.g2(bintime=1e-3, y_min=1e9)
    # the lifetime analysis state is untouched
    assert p.selected_channels is None and p._tt_cache == {}


def test_picoharp_t2_needs_the_period_from_the_user(tmp_path):
    rng = np.random.default_rng(5)
    events, _ = antibunching_events(rng, period=PERIOD * 4, n_pulses=100000, single=True,
                                    brightness=0.1, sync=False)
    # PicoHarp 300 T2: 4 ps time tags; here 10 ps ticks for the same events
    path = write_ptu(tmp_path / "ph.ptu", encode_t2(events, "picoharp"),
                     record_type=PICOHARP_T2, global_resolution=TICK, acquisition_ms=40)
    p = PTU(path)
    assert not p.has_lifetime
    continuous = p.g2(max_lag=1e-6)
    assert continuous.meta["method"] == "continuous" and continuous.meta["period"] is None
    pulsed = p.g2(period=PERIOD * 4 * TICK)
    assert pulsed.meta["method"] == "pulsed" and pulsed.meta["g2_0"] < 0.1


def test_g2_rejects_t3_files_and_bad_channels(tmp_path):
    t3 = PTU(write_ptu(tmp_path / "t3.ptu",
                       encode_t3([(5, "photon", 1, 100), (9, "photon", 2, 7)], "generic"),
                       record_type=GENERIC_T3, global_resolution=25e-9, resolution=5e-12))
    with pytest.raises(ValueError, match="T2 file"):
        t3.g2()
    rng = np.random.default_rng(6)
    p, _ = _multiharp(tmp_path, rng, n_pulses=20000)
    with pytest.raises(ValueError, match="no photons on channel"):
        p.g2(channels=(1, 3))
    with pytest.raises(ValueError, match="two different"):
        p.g2(channels=(1, 1))
