"""Security regression tests for the untrusted-file PTU parser."""

import io
import struct

import pytest

from kpfm_pl.ptu import PTU


def _buf(length: int, payload: bytes = b"") -> io.BytesIO:
    return io.BytesIO(struct.pack("<q", length) + payload)


def test_length_prefixed_rejects_oversized_length():
    # A tiny file that claims a multi-gigabyte string must not be allocated.
    fh = _buf(2**40, b"abc")
    with pytest.raises(ValueError):
        PTU._read_length_prefixed(fh)


def test_length_prefixed_rejects_negative_length():
    fh = _buf(-1, b"abc")
    with pytest.raises(ValueError):
        PTU._read_length_prefixed(fh)


def test_length_prefixed_reads_valid_payload():
    fh = _buf(3, b"abc")
    assert PTU._read_length_prefixed(fh) == b"abc"
