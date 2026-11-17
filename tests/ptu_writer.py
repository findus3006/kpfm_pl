"""Reference writers of synthetic PTU files for the tests (not a test module).

Small record encoders that follow PicoQuant's ``Read_PTU.py`` (bit layouts,
overflow records, markers, sync events), so the vectorised decoders of
:mod:`kpfm_pl.ptu` are checked against known events.
"""

import struct
from pathlib import Path

import numpy as np

from kpfm_pl.ptu import _TAG_TYPES

PICOHARP_T3, PICOHARP_T2 = 0x00010303, 0x00010203
HYDRA1_T3, HYDRA1_T2 = 0x00010304, 0x00010204
GENERIC_T3, GENERIC_T2 = 0x00010307, 0x00010207

T2_WRAP = {("picoharp", 1): 210698240, ("generic", 1): 33552000, ("generic", 2): 33554432}
T3_WRAP = {"picoharp": 65536, "generic": 1024}


# ------------------------------------------------------------------ encoders
def _tag(name: str, typ: str, value: bytes = b"\x00" * 8) -> bytes:
    return name.encode().ljust(32, b"\x00") + struct.pack("<ii", -1, _TAG_TYPES[typ]) + value


def write_ptu(path: Path, records: np.ndarray, *, record_type: int | None,
              global_resolution: float, resolution: float | None = None,
              acquisition_ms: int = 1000, sync_rate: int | None = None,
              declared: int | None = None) -> Path:
    """Write a minimal PTU file holding ``records`` (uint32)."""
    records = np.asarray(records, dtype="<u4")
    n = len(records) if declared is None else declared
    data = b"PQTTTR\x00\x00" + b"1.0.00\x00\x00"
    data += _tag("TTResult_NumberOfRecords", "Int8", struct.pack("<q", n))
    data += _tag("MeasDesc_GlobalResolution", "Float8", struct.pack("<d", global_resolution))
    if resolution is not None:
        data += _tag("MeasDesc_Resolution", "Float8", struct.pack("<d", resolution))
    data += _tag("MeasDesc_AcquisitionTime", "Int8", struct.pack("<q", acquisition_ms))
    if record_type is not None:
        data += _tag("TTResultFormat_TTTRRecType", "Int8", struct.pack("<q", record_type))
    if sync_rate is not None:
        data += _tag("TTResult_SyncRate", "Int8", struct.pack("<q", sync_rate))
    data += _tag("Header_End", "Empty8")
    path.write_bytes(data + records.tobytes())
    return path


def encode_t2(events, layout: str, version: int = 2) -> np.ndarray:
    """Encode time-sorted ``(tick, kind, value)`` events as T2 records.

    ``kind`` is ``"photon"`` (``value`` = detector, 1-based), ``"sync"`` or
    ``"marker"`` (``value`` = marker bits). Overflow records are inserted
    where the time tag wraps (a counted one for the generic V2 layout).
    """
    wrap = T2_WRAP[(layout, 1 if layout == "picoharp" else version)]
    out, overflows = [], 0
    for tick, kind, value in events:
        missing = int(tick) // wrap - overflows
        if missing > 0:
            if layout == "picoharp":
                out += [0xF0000000] * missing
            elif version == 1:
                out += [(1 << 31) | (0x3F << 25)] * missing
            else:
                out.append((1 << 31) | (0x3F << 25) | missing)
            overflows += missing
        t = int(tick) - overflows * wrap
        if layout == "picoharp":
            if kind == "photon":
                out.append(((value - 1) << 28) | t)
            elif kind == "marker":
                out.append((0xF << 28) | (t & ~0xF) | value)
            else:
                raise ValueError("the PicoHarp 300 records no sync events in T2")
        elif kind == "photon":
            out.append(((value - 1) << 25) | t)
        elif kind == "sync":
            out.append((1 << 31) | t)
        else:
            out.append((1 << 31) | (value << 25) | t)
    return np.array(out, dtype="<u4")


def encode_t3(events, layout: str, version: int = 2) -> np.ndarray:
    """Encode sync-sorted ``(nsync, kind, value, dtime)`` events as T3 records."""
    wrap = T3_WRAP[layout]
    out, overflows = [], 0
    for nsync, kind, value, dtime in events:
        missing = int(nsync) // wrap - overflows
        if missing > 0:
            if layout == "picoharp":
                out += [0xF0000000] * missing
            elif version == 1:
                out += [(1 << 31) | (0x3F << 25)] * missing
            else:
                out.append((1 << 31) | (0x3F << 25) | missing)
            overflows += missing
        n = int(nsync) - overflows * wrap
        if layout == "picoharp":
            if kind == "photon":
                out.append((value << 28) | (dtime << 16) | n)
            else:
                out.append((0xF << 28) | (value << 16) | n)
        elif kind == "photon":
            out.append(((value - 1) << 25) | (dtime << 10) | n)
        else:
            out.append((1 << 31) | (value << 25) | n)
    return np.array(out, dtype="<u4")


def decay_events(rng, *, period=20000, n_pulses=40000, n_photons=30000, tau=600.0,
                 rise=2000, divider=1, first_pulse=0):
    """Laser pulses every ``period`` ticks and photons decaying with ``tau`` ticks.

    Every ``divider``-th pulse from ``first_pulse`` on is recorded as a sync
    event. Returns ``(events, photons)`` with ``photons`` the sorted array
    ``(tick, detector, delay after the pulse)``.
    """
    pulse = rng.integers(0, n_pulses, n_photons)
    delay = (rise + rng.exponential(tau, n_photons).astype(np.int64)) % period
    tick = pulse * period + delay
    detector = rng.integers(1, 3, n_photons)
    order = np.argsort(tick, kind="stable")
    photons = np.column_stack((tick[order], detector[order], delay[order]))
    sync = [(int(k) * period, "sync", 0) for k in range(first_pulse, n_pulses, divider)]
    events = sync + [(int(t), "photon", int(d)) for t, d, _ in photons]
    # sync events first at equal times, as the hardware would report them
    events.sort(key=lambda event: (event[0], event[1] != "sync"))
    return events, photons


def antibunching_events(rng, *, period=10000, n_pulses=200000, brightness=0.05,
                        single=True, tau=500.0, rise=1000, background=0.0, sync=True,
                        dark_pulses=None):
    """Pulsed light on a 50/50 beam splitter with two detectors (channels 1 and 2).

    Every laser pulse (every ``period`` ticks) yields photons ``tau`` ticks
    (exponential) after ``rise``: with ``single`` a single-photon emitter (at
    most one photon per pulse, with probability ``brightness``), otherwise
    Poisson light with mean ``brightness`` photons per pulse. Each photon goes
    to detector 1 or 2 at random. ``background`` adds uncorrelated counts per
    pulse and detector. Pulses in ``dark_pulses`` (a boolean mask, e.g. a
    blinking off state) emit nothing. With ``sync`` every pulse is recorded
    as a sync event. Returns ``(events, ticks)`` with ``ticks`` = ``{1: ..., 2: ...}``.
    """
    if single:
        n = (rng.random(n_pulses) < brightness).astype(np.int64)
    else:
        n = rng.poisson(brightness, n_pulses)
    if dark_pulses is not None:
        n[np.asarray(dark_pulses, dtype=bool)] = 0
    pulse = np.repeat(np.arange(n_pulses), n)
    tick = pulse * period + rise + rng.exponential(tau, pulse.size).astype(np.int64)
    detector = rng.integers(1, 3, pulse.size)
    if background:
        n_bg = rng.poisson(background * n_pulses * 2)
        tick = np.concatenate((tick, rng.integers(0, n_pulses * period, n_bg)))
        detector = np.concatenate((detector, rng.integers(1, 3, n_bg)))
    order = np.argsort(tick, kind="stable")
    tick, detector = tick[order], detector[order]
    # two photons cannot share a time tag on one detector (dead time)
    keep = np.ones(tick.size, dtype=bool)
    for d in (1, 2):
        idx = np.flatnonzero(detector == d)
        keep[idx[1:][np.diff(tick[idx]) == 0]] = False
    tick, detector = tick[keep], detector[keep]
    events = [(int(t), "photon", int(d)) for t, d in zip(tick, detector)]
    if sync:
        events += [(int(k) * period, "sync", 0) for k in range(n_pulses)]
        events.sort(key=lambda event: (event[0], event[1] != "sync"))
    return events, {d: tick[detector == d] for d in (1, 2)}
