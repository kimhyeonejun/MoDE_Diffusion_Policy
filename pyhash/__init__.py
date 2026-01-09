"""
Compatibility shim for the `pyhash` package.

The original `pyhash` (pyfasthash) project requires building a C/C++ extension and
does not reliably build on modern Python versions (e.g., 3.12). This repo only
needs `pyhash.fnv1_32()` to create deterministic seeds in evaluation utilities.

This module provides a small subset of the API:

    import pyhash
    hasher = pyhash.fnv1_32()
    seed = hasher("some string")   # -> int in [0, 2**32)
"""

from __future__ import annotations

from typing import Callable, Union

_BytesLike = Union[bytes, bytearray, memoryview]
_StrOrBytes = Union[str, _BytesLike]


def _to_bytes(x: _StrOrBytes) -> bytes:
    if isinstance(x, str):
        return x.encode("utf-8", errors="surrogatepass")
    if isinstance(x, (bytes, bytearray, memoryview)):
        return bytes(x)
    return str(x).encode("utf-8", errors="surrogatepass")


def fnv1_32() -> Callable[[_StrOrBytes], int]:
    """Return a callable implementing FNV-1 32-bit."""

    FNV_OFFSET_BASIS = 2166136261
    FNV_PRIME = 16777619
    MOD_MASK = 0xFFFFFFFF

    def _hash(data: _StrOrBytes) -> int:
        h = FNV_OFFSET_BASIS
        for b in _to_bytes(data):
            h = (h * FNV_PRIME) & MOD_MASK
            h = (h ^ b) & MOD_MASK
        return h

    return _hash


def fnv1a_32() -> Callable[[_StrOrBytes], int]:
    """Return a callable implementing FNV-1a 32-bit (compatibility)."""

    FNV_OFFSET_BASIS = 2166136261
    FNV_PRIME = 16777619
    MOD_MASK = 0xFFFFFFFF

    def _hash(data: _StrOrBytes) -> int:
        h = FNV_OFFSET_BASIS
        for b in _to_bytes(data):
            h = (h ^ b) & MOD_MASK
            h = (h * FNV_PRIME) & MOD_MASK
        return h

    return _hash


__all__ = ["fnv1_32", "fnv1a_32"]


