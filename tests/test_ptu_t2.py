"""Record-format dispatch and T2 decoding of the PTU reader, on synthetic files.

The files are written record by record with the reference encoders of
``ptu_writer.py``, so the vectorised decoders are checked against known events.
"""

import numpy as np
import pytest
from ptu_writer import (
    GENERIC_T2,
    GENERIC_T3,
    HYDRA1_T2,
    HYDRA1_T3,
    PICOHARP_T2,
    PICOHARP_T3,
    T2_WRAP,
    decay_events,
    encode_t2,
    encode_t3,
    write_ptu,
)

from kpfm_pl.ptu import MAX_LIFETIME_BINS, PTU, RECORD_TYPES, _decode_records


# ------------------------------------------------------------------ dispatch
def test_every_record_type_of_the_reference_reader_is_known():
    assert set(RECORD_TYPES) == {
        0x00010303, 0x00010203, 0x00010304, 0x00010204, 0x01010304, 0x01010204,
        0x00010305, 0x00010205, 0x00010306, 0x00010206, 0x00010307, 0x00010207}
    assert {mode for mode, *_ in RECORD_TYPES.values()} == {"T2", "T3"}
    # only the first HydraHarp firmware uses the old overflow records
    assert [key for key, value in RECORD_TYPES.items() if value[1:3] == ("generic", 1)] == [
        0x00010304, 0x00010204]


def test_unknown_record_type_is_rejected(tmp_path):
    path = write_ptu(tmp_path / "odd.ptu", [], record_type=0x00010399,
                     global_resolution=1e-7, resolution=1e-10)
    with pytest.raises(ValueError, match="unsupported TTTR record type 0x00010399"):
        PTU(path)


def test_file_without_record_type_is_read_as_picoharp_t3(tmp_path):
    events = [(5, "photon", 1, 100), (70000, "marker", 4, 0), (70010, "photon", 2, 7)]
    records = encode_t3(events, "picoharp")
    for record_type in (None, PICOHARP_T3):
        p = PTU(write_ptu(tmp_path / "t3.ptu", records, record_type=record_type,
                          global_resolution=1e-7, resolution=1e-10))
        assert (p.measurement_mode, p.device, p.has_lifetime) == ("T3", "PicoHarp 300", True)
        np.testing.assert_array_equal(p.events[:, 0], [5, 70010])     # one overflow passed
        np.testing.assert_array_equal(p.events[:, 1], [100, 7])
        np.testing.assert_allclose(p.events[:, 2], np.array([5, 70010]) * 1e-7)
        assert p.marker == {4: [70000]}
        assert p.channels == (1, 2) and p.photon_channel.tolist() == [1, 2]
        assert p.n_lt_bins == int(1e-7 / 1e-10) + 1 and p.lifetime_binning == 1
        np.testing.assert_allclose(p.photon_times(), [5e-7 + 100e-10, 70010e-7 + 7e-10])


@pytest.mark.parametrize("record_type, version", [(HYDRA1_T3, 1), (GENERIC_T3, 2)])
def test_generic_t3_records(tmp_path, record_type, version):
    # 25 ns period at 5 ps: 5000 native bins, rebinned by 2 to 2500 lifetime bins
    events = [(3, "photon", 1, 40), (1500, "photon", 2, 4999), (3000, "marker", 4, 0),
              (9000, "photon", 2, 2001), (9001, "photon", 1, 5200)]
    records = encode_t3(events, "generic", version)
    if version == 2:
        assert len(records) == len(events) + 3       # counted overflow records
    p = PTU(write_ptu(tmp_path / "t3.ptu", records, record_type=record_type,
                      global_resolution=25e-9, resolution=5e-12))
    assert p.measurement_mode == "T3" and p.has_lifetime
    assert p.lifetime_binning == 2 and p.resolution == pytest.approx(10e-12)
    assert p.n_lt_bins == int(25e-9 / p.resolution) + 1 <= MAX_LIFETIME_BINS
    # the photon past the period (dtime 5200 -> bin 2600) is dropped, as for the PicoHarp
    np.testing.assert_array_equal(p.events[:, 0], [3, 1500, 9000])
    np.testing.assert_array_equal(p.events[:, 1], [20, 2499, 1000])
    assert p.photon_channel.tolist() == [1, 2, 2] and p.channels == (1, 2)
    assert p.marker == {4: [3000]}
    np.testing.assert_allclose(p.get_markers(4), [3000 * 25e-9])
    native = PTU(write_ptu(tmp_path / "t3.ptu", records, record_type=record_type,
                           global_resolution=25e-9, resolution=5e-12), lifetime_binning=1)
    assert native.n_lt_bins == int(25e-9 / 5e-12) + 1 > MAX_LIFETIME_BINS
    assert native.events[:, 1].tolist() == [40, 4999, 2001]


def test_decode_records_carries_the_overflow_between_blocks():
    events = [(k * 9_000_000, "photon", 1 + k % 2) for k in range(40)]
    records = encode_t2(events, "generic", 2)
    whole = _decode_records(records, "T2", "generic", 2, 0)
    np.testing.assert_array_equal(whole["counter"], [tick for tick, *_ in events])
    first = _decode_records(records[:17], "T2", "generic", 2, 0)
    second = _decode_records(records[17:], "T2", "generic", 2, first["overflow"])
    np.testing.assert_array_equal(
        np.concatenate((first["counter"], second["counter"])), whole["counter"])
    with pytest.raises(ValueError, match="layout"):
        _decode_records(records, "T2", "other", 2, 0)


# ------------------------------------------------------- T2 without a laser
@pytest.fixture
def picoharp_t2(tmp_path):
    """PicoHarp 300 T2: two detectors, 4 ps time tags, two laser-trigger markers."""
    rng = np.random.default_rng(1)
    ticks = np.sort(rng.integers(0, int(1.0 / 4e-12), 4000))        # 1 s, many overflows
    detectors = rng.integers(1, 3, ticks.size)
    markers = [int(0.25 / 4e-12) // 16 * 16, int(0.75 / 4e-12) // 16 * 16]
    events = [(int(t), "photon", int(d)) for t, d in zip(ticks, detectors)]
    events += [(m, "marker", 4) for m in markers]
    events.sort(key=lambda event: event[0])
    path = write_ptu(tmp_path / "ph_t2.ptu", encode_t2(events, "picoharp"),
                     record_type=PICOHARP_T2, global_resolution=4e-12)
    return path, ticks, detectors, markers


def test_picoharp_t2_time_tags_and_channels(picoharp_t2):
    path, ticks, detectors, markers = picoharp_t2
    p = PTU(path)
    assert (p.measurement_mode, p.device) == ("T2", "PicoHarp 300")
    assert p.has_lifetime is False and p.is_cw is False
    assert p.timetag_resolution == 4e-12 and p.sync_events == 0
    assert p.channels == (1, 2)                     # the inputs 0 and 1 of the file
    np.testing.assert_array_equal(p.photon_channel, detectors)
    np.testing.assert_allclose(p.photon_times(), ticks * 4e-12, rtol=0, atol=1e-15)
    for detector in (1, 2):
        np.testing.assert_allclose(p.photon_times(detector),
                                   ticks[detectors == detector] * 4e-12, rtol=0, atol=1e-15)
    events = p.get_events()
    np.testing.assert_array_equal(events[:, 0], ticks)              # the time tag itself
    assert not events[:, 1].any() and p.n_lt_bins == 1
    # the marker bits sit in the lowest 4 bits of the time tag (at most 15 ticks off)
    np.testing.assert_allclose(p.get_markers(4), np.array(markers) * 4e-12, atol=16 * 4e-12)


def test_picoharp_t2_time_trace_triggers_and_channel_selection(picoharp_t2):
    path, ticks, detectors, _ = picoharp_t2
    p = PTU(path)
    trace = p.time_trace(bintime=0.05)
    expected = np.bincount((ticks * 4e-12 / 0.05).astype(int), minlength=20)
    np.testing.assert_array_equal(trace.y, expected)
    assert trace.meta["matrix"].shape == (20, 1)
    triggers = p.laser_triggers()
    assert [point["index"] for point in triggers] == [5, 15]
    assert [point["time"] for point in triggers] == [0.25, 0.75]

    assert p.set_channels(2) == (2,) and p.selected_channels == (2,)
    single = p.time_trace(bintime=0.05)
    np.testing.assert_array_equal(
        single.y, np.bincount((ticks[detectors == 2] * 4e-12 / 0.05).astype(int), minlength=20))
    assert len(p.get_events()) == np.count_nonzero(detectors == 2)
    assert p.laser_triggers() == triggers           # markers do not depend on the channel
    assert p.set_channels([1, 2]) == (1, 2)
    np.testing.assert_array_equal(p.time_trace(bintime=0.05).y, expected)
    assert p.set_channels(None) == (1, 2) and p.selected_channels is None
    assert PTU(path, channels=1).selected_channels == (1,)
    with pytest.raises(ValueError, match=r"no photons on channel\(s\) \[3\]"):
        p.set_channels(3)
    with pytest.raises(ValueError, match="detector numbers"):
        p.set_channels(1.5)


def test_t2_without_laser_pulses_rejects_lifetime_analysis(picoharp_t2):
    p = PTU(picoharp_t2[0])
    p.time_trace(bintime=0.05)
    hist = None
    for call in (p.lifetime_histogram, p.threshold, p.lifetime_trace, p.arrival_time_trace,
                 p.flid, lambda: p.set_lifetime_shift("auto"),
                 lambda: p.fit_lifetime(hist), lambda: p.irf_histogram(np.ones(1))):
        with pytest.raises(ValueError, match="no lifetime information"):
            call()
    assert not p.seam_bins().any()
    with pytest.raises(ValueError, match="no lifetime information"):
        PTU(picoharp_t2[0], lifetime_shift="auto")


@pytest.mark.parametrize("version, record_type", [(1, HYDRA1_T2), (2, GENERIC_T2)])
def test_generic_t2_overflows_markers_and_missing_sync(tmp_path, version, record_type):
    # sparse photons: several time-tag wraparounds lie between two records
    ticks = np.array([10, 33_000_000, 250_000_000, 250_000_001, 1_900_000_000])
    events = [(int(t), "photon", 1 + k % 2) for k, t in enumerate(ticks)]
    events.insert(3, (250_000_000, "marker", 4))
    records = encode_t2(events, "generic", version)
    wraps = int(ticks[-1]) // T2_WRAP[("generic", version)]
    # V1 writes one record per wraparound, V2 one counted record per gap
    assert wraps > 50 and len(records) == len(events) + (wraps if version == 1 else 2)
    p = PTU(write_ptu(tmp_path / "t2.ptu", records, record_type=record_type,
                      global_resolution=5e-12))
    assert p.measurement_mode == "T2" and p.has_lifetime is False
    np.testing.assert_array_equal(p.get_events()[:, 0], ticks)
    assert p.photon_channel.tolist() == [1, 2, 1, 2, 1]
    np.testing.assert_allclose(p.get_markers(4), [250_000_000 * 5e-12])


def test_truncated_t2_file_warns_and_reads_what_is_there(tmp_path):
    events = [(k * 1000, "photon", 1) for k in range(10)]
    path = write_ptu(tmp_path / "cut.ptu", encode_t2(events, "picoharp"),
                     record_type=PICOHARP_T2, global_resolution=4e-12, declared=25)
    with pytest.warns(RuntimeWarning, match="file ended earlier than expected"):
        p = PTU(path)
    assert len(p.get_events()) == 10
    with pytest.raises(ValueError, match="invalid number of records"):
        PTU(write_ptu(tmp_path / "neg.ptu", [], record_type=PICOHARP_T2,
                      global_resolution=4e-12, declared=-3))
    with pytest.raises(ValueError, match="invalid time-tag resolution"):
        PTU(write_ptu(tmp_path / "inf.ptu", [], record_type=PICOHARP_T2,
                      global_resolution=np.inf))


# --------------------------------------------------------- T2 with a laser
def test_t2_with_laser_sync_recovers_the_t3_information(tmp_path):
    rng = np.random.default_rng(2)
    events, photons = decay_events(rng)             # 100 ns period at 5 ps, tau = 3 ns
    path = write_ptu(tmp_path / "mh_t2.ptu", encode_t2(events, "generic"),
                     record_type=GENERIC_T2, global_resolution=5e-12,
                     acquisition_ms=4, sync_rate=10_000_000)
    p = PTU(path)
    assert p.measurement_mode == "T2" and p.has_lifetime and not p.is_cw
    assert p.device == "MultiHarp / PicoHarp 330"
    assert p.sync_events == 40000 and p.sync_divider == 1
    assert p.rep_time == pytest.approx(100e-9, rel=1e-9)
    # 20000 time-tag units per period -> 8 units per lifetime bin
    assert p.lifetime_binning == 8 and p.resolution == pytest.approx(40e-12)
    assert p.n_lt_bins == 2501 <= MAX_LIFETIME_BINS
    assert p.channels == (1, 2)
    np.testing.assert_array_equal(p.photon_channel, photons[:, 1])

    events_table = p.get_events()
    np.testing.assert_array_equal(events_table[:, 1], photons[:, 2] // 8)   # delay after pulse
    np.testing.assert_array_equal(events_table[:, 0], photons[:, 0] // 20000)   # pulse number
    np.testing.assert_allclose(events_table[:, 2], photons[:, 0] * 5e-12, rtol=0, atol=1e-15)
    np.testing.assert_allclose(p.photon_times(2),
                               photons[photons[:, 1] == 2, 0] * 5e-12, rtol=0, atol=1e-15)

    trace = p.time_trace(bintime=1e-3)
    assert trace.y.sum() == len(photons)
    np.testing.assert_array_equal(
        trace.y, np.bincount((photons[:, 0] * 5e-12 / 1e-3).astype(int), minlength=4))
    hist = p.lifetime_histogram()
    assert hist.y.sum() == len(photons) and len(hist) == 2501
    assert hist.x[np.argmax(hist.y)] == pytest.approx(10e-9, abs=0.3e-9)    # rise at 2000 ticks
    fit = p.fit_lifetime(hist, n_exp=1, weighting="mle")
    assert fit["tau"][0] == pytest.approx(3.0, rel=0.05)
    assert p.lifetime_trace(bintime=1e-3, min_counts=None).y == pytest.approx(3.0, rel=0.1)

    # one detector at a time, and a lifetime shift that follows the selection
    p.set_channels(1)
    assert p.lifetime_histogram().y.sum() == np.count_nonzero(photons[:, 1] == 1)
    shift = p.set_lifetime_shift(5.0)
    assert shift == pytest.approx(5.0, abs=0.04)
    peak = p.lifetime_histogram().x[np.argmax(p.lifetime_histogram().y)]
    assert peak == pytest.approx(5e-9, abs=0.3e-9)
    p.set_channels(None)                            # the shift is applied to all photons again
    assert p.lifetime_shift_ns == shift and len(p.get_events()) == len(photons)
    assert p.lifetime_histogram().x[np.argmax(p.lifetime_histogram().y)] == pytest.approx(
        5e-9, abs=0.3e-9)
    p.set_lifetime_shift(None)
    np.testing.assert_array_equal(p.get_events()[:, 1], photons[:, 2] // 8)


def test_t2_sync_divider_is_folded_back_to_the_laser_period(tmp_path):
    rng = np.random.default_rng(3)
    events, photons = decay_events(rng, divider=8)  # every 8th pulse recorded
    records = encode_t2(events, "generic")
    path = write_ptu(tmp_path / "div.ptu", records, record_type=GENERIC_T2,
                     global_resolution=5e-12, acquisition_ms=4, sync_rate=10_000_000)
    p = PTU(path)                                   # divider from the header sync rate
    assert p.sync_events == 5000 and p.sync_divider == 8
    assert p.rep_time == pytest.approx(100e-9, rel=1e-9) and p.n_lt_bins == 2501
    np.testing.assert_array_equal(p.get_events()[:, 1], photons[:, 2] // 8)
    # without the header rate the recorded interval counts as the period ...
    bare = write_ptu(tmp_path / "bare.ptu", records, record_type=GENERIC_T2,
                     global_resolution=5e-12, acquisition_ms=4)
    undivided = PTU(bare)
    assert undivided.sync_divider == 1 and undivided.rep_time == pytest.approx(800e-9, rel=1e-9)
    # ... unless the divider is given
    given = PTU(bare, sync_divider=8)
    assert given.rep_time == pytest.approx(100e-9, rel=1e-9)
    np.testing.assert_array_equal(given.get_events()[:, 1], photons[:, 2] // 8)
    # a rate that does not fit a whole number of pulses is not trusted
    odd = write_ptu(tmp_path / "odd.ptu", records, record_type=GENERIC_T2,
                    global_resolution=5e-12, acquisition_ms=4, sync_rate=4_300_000)
    with pytest.warns(RuntimeWarning, match="assuming every pulse was recorded"):
        assert PTU(odd).sync_divider == 1


def test_t2_photons_before_the_first_and_after_missing_sync_events(tmp_path):
    rng = np.random.default_rng(4)
    events, photons = decay_events(rng, n_pulses=20000, n_photons=20000, first_pulse=500)
    # the laser sync drops out for a while: the delays fold back by whole periods
    events = [event for event in events
              if not (event[1] == "sync" and 8000 <= event[0] // 20000 < 8400)]
    path = write_ptu(tmp_path / "gap.ptu", encode_t2(events, "generic"),
                     record_type=GENERIC_T2, global_resolution=5e-12, acquisition_ms=2)
    p = PTU(path)
    assert p.sync_events == 20000 - 500 - 400
    assert p.rep_time == pytest.approx(100e-9, rel=1e-9)
    assert len(p.get_events()) == len(photons)      # no photon is lost
    assert np.count_nonzero(photons[:, 0] < 500 * 20000) > 100
    np.testing.assert_array_equal(p.get_events()[:, 1], photons[:, 2] // 8)


def test_t2_sync_that_is_not_periodic_gives_no_lifetimes(tmp_path):
    rng = np.random.default_rng(5)
    sync = np.sort(rng.integers(0, 10**9, 300))
    events = [(int(t), "sync", 0) for t in sync] + [(k * 10**6, "photon", 1) for k in range(900)]
    events.sort(key=lambda event: event[0])
    path = write_ptu(tmp_path / "noise.ptu", encode_t2(events, "generic"),
                     record_type=GENERIC_T2, global_resolution=5e-12, acquisition_ms=5)
    with pytest.warns(RuntimeWarning, match="not periodic"):
        p = PTU(path)
    assert p.has_lifetime is False and p.sync_events == 300
    assert p.time_trace(bintime=1e-3).y.sum() == 900


def test_lifetime_binning_can_be_chosen(tmp_path):
    rng = np.random.default_rng(6)
    events, photons = decay_events(rng, n_pulses=4000, n_photons=3000)
    path = write_ptu(tmp_path / "bin.ptu", encode_t2(events, "generic"),
                     record_type=GENERIC_T2, global_resolution=5e-12, acquisition_ms=1)
    p = PTU(path, lifetime_binning=16)
    assert p.resolution == pytest.approx(80e-12) and p.n_lt_bins == 1251
    np.testing.assert_array_equal(p.get_events()[:, 1], photons[:, 2] // 16)
    for name in ("lifetime_binning", "sync_divider"):
        for bad in (0, -2, 1.5, True):
            with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
                PTU(path, **{name: bad})


# ------------------------------------------------------- T3 channel selection
def test_t3_channel_selection_keeps_the_raw_data(tmp_path):
    rng = np.random.default_rng(7)
    nsync = np.sort(rng.integers(0, 200000, 5000))
    events = [(int(n), "photon", 1 + k % 2, int(rng.integers(0, 1000)))
              for k, n in enumerate(nsync)]
    path = write_ptu(tmp_path / "two.ptu", encode_t3(events, "picoharp"),
                     record_type=PICOHARP_T3, global_resolution=1e-7, resolution=1e-10,
                     acquisition_ms=20)
    p = PTU(path)
    assert p.channels == (1, 2) and p.selected_channels is None
    full = p.get_events().copy()
    total = p.time_trace(bintime=5e-3).y
    p.set_channels(2)
    np.testing.assert_array_equal(p.get_events(), full[1::2])
    part = p.time_trace(bintime=5e-3).y
    assert part.sum() == 2500 and np.all(part <= total)
    p.set_lifetime_shift(20.0)
    p.set_channels(None)                            # shift re-applied to the full table
    np.testing.assert_array_equal(p.get_events()[:, 1], (full[:, 1] - 200) % 1000)
    p.set_lifetime_shift(None)
    np.testing.assert_array_equal(p.get_events(), full)
    np.testing.assert_allclose(p.photon_times(1), full[0::2, 0] * 1e-7 + full[0::2, 1] * 1e-10)
