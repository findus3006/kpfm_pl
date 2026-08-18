"""Security regression tests for the untrusted-file PTU parser."""

import io
import struct

import numpy as np
import pytest

from kpfm_pl.ptu import PTU, _TAG_TYPES


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


def test_float_array_is_consumed_and_decoded():
    payload = np.array([1.5, -2.0], dtype="<f8").tobytes()
    fh = _buf(len(payload), payload + b"next")
    value = PTU._read_tag_value(fh, _TAG_TYPES["Float8Array"])
    np.testing.assert_allclose(value, [1.5, -2.0])
    assert fh.read() == b"next"


def test_float_array_rejects_non_float_payload_length():
    with pytest.raises(ValueError, match="multiple of 8"):
        PTU._read_tag_value(_buf(3, b"abc"), _TAG_TYPES["Float8Array"])


@pytest.mark.parametrize("length", [-1, 100])
def test_binary_blob_rejects_invalid_length(length):
    with pytest.raises(ValueError, match="payload length"):
        PTU._read_tag_value(_buf(length, b"abc"), _TAG_TYPES["BinaryBlob"])


def test_binary_blob_is_consumed():
    fh = _buf(3, b"abcnext")
    assert PTU._read_tag_value(fh, _TAG_TYPES["BinaryBlob"]) == b"abc"
    assert fh.read() == b"next"


def test_fixed_size_tag_rejects_truncated_value():
    with pytest.raises(ValueError, match="Truncated PTU integer tag value"):
        PTU._read_tag_value(io.BytesIO(b"\0" * 4), _TAG_TYPES["Int8"])


def test_header_rejects_truncated_fixed_size_tag_header():
    fh = io.BytesIO(b"PQTTTR\0\0" + b"1.0\0\0\0\0\0" + b"short")
    ptu = PTU.__new__(PTU)
    with pytest.raises(ValueError, match="Truncated PTU tag identifier"):
        ptu._read_header(fh)
