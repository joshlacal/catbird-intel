"""Pure-Python CityHash64 (v1.0.2), the variant ClickHouse's cityHash64(String) uses.

This is the Mac-side reference for the S8 cross-check: the VM prints 1,000
(string, cityHash64) pairs from its clickhouse-local, and crosscheck_hash.py
recomputes them here and with the Mac's clickhouse. Every sample predicate in
the pass (`cityHash64(did) % 200 == 0`, `% 16`, `% 50 == 7`, ...) depends on the
two agreeing bit for bit.
"""
from __future__ import annotations

import struct

MASK = (1 << 64) - 1
K0 = 0xC3A5C85C97CB3127
K1 = 0xB492B66FBE98F273
K2 = 0x9AE16A3B2F90404F
K3 = 0xC949D7C7509E6557
KMUL = 0x9DDFEA08EB382D69


def _f64(s: bytes, i: int) -> int:
    return struct.unpack_from("<Q", s, i)[0]


def _f32(s: bytes, i: int) -> int:
    return struct.unpack_from("<I", s, i)[0]


def _rot(v: int, shift: int) -> int:
    if shift == 0:
        return v
    return ((v >> shift) | (v << (64 - shift))) & MASK


def _rot1(v: int, shift: int) -> int:  # RotateByAtLeast1
    return ((v >> shift) | (v << (64 - shift))) & MASK


def _shift_mix(v: int) -> int:
    return v ^ (v >> 47)


def _hash128to64(u: int, v: int) -> int:
    a = ((u ^ v) * KMUL) & MASK
    a ^= a >> 47
    b = ((v ^ a) * KMUL) & MASK
    b ^= b >> 47
    return (b * KMUL) & MASK


def _len0to16(s: bytes, n: int) -> int:
    if n > 8:
        a = _f64(s, 0)
        b = _f64(s, n - 8)
        return _hash128to64(a, _rot1((b + n) & MASK, n)) ^ b
    if n >= 4:
        a = _f32(s, 0)
        return _hash128to64((n + (a << 3)) & MASK, _f32(s, n - 4))
    if n > 0:
        a, b, c = s[0], s[n >> 1], s[n - 1]
        y = (a + (b << 8)) & 0xFFFFFFFF
        z = (n + (c << 2)) & 0xFFFFFFFF
        return (_shift_mix(((y * K2) ^ (z * K3)) & MASK) * K2) & MASK
    return K2


def _len17to32(s: bytes, n: int) -> int:
    a = (_f64(s, 0) * K1) & MASK
    b = _f64(s, 8)
    c = (_f64(s, n - 8) * K2) & MASK
    d = (_f64(s, n - 16) * K0) & MASK
    return _hash128to64(
        (_rot((a - b) & MASK, 43) + _rot(c, 30) + d) & MASK,
        (a + _rot(b ^ K3, 20) - c + n) & MASK,
    )


def _weak32(w: int, x: int, y: int, z: int, a: int, b: int) -> tuple[int, int]:
    a = (a + w) & MASK
    b = _rot((b + a + z) & MASK, 21)
    c = a
    a = (a + x) & MASK
    a = (a + y) & MASK
    b = (b + _rot(a, 44)) & MASK
    return (a + z) & MASK, (b + c) & MASK


def _weak32s(s: bytes, i: int, a: int, b: int) -> tuple[int, int]:
    return _weak32(_f64(s, i), _f64(s, i + 8), _f64(s, i + 16), _f64(s, i + 24), a, b)


def _len33to64(s: bytes, n: int) -> int:
    z = _f64(s, 24)
    a = (_f64(s, 0) + (n + _f64(s, n - 16)) * K0) & MASK
    b = _rot((a + z) & MASK, 52)
    c = _rot(a, 37)
    a = (a + _f64(s, 8)) & MASK
    c = (c + _rot(a, 7)) & MASK
    a = (a + _f64(s, 16)) & MASK
    vf = (a + z) & MASK
    vs = (b + _rot(a, 31) + c) & MASK
    a = (_f64(s, 16) + _f64(s, n - 32)) & MASK
    z = _f64(s, n - 8)
    b = _rot((a + z) & MASK, 52)
    c = _rot(a, 37)
    a = (a + _f64(s, n - 24)) & MASK
    c = (c + _rot(a, 7)) & MASK
    a = (a + _f64(s, n - 16)) & MASK
    wf = (a + z) & MASK
    ws = (b + _rot(a, 31) + c) & MASK
    r = _shift_mix(((vf + ws) * K2 + (wf + vs) * K0) & MASK)
    return (_shift_mix((r * K0 + vs) & MASK) * K2) & MASK


def city_hash64(data: bytes | str) -> int:
    s = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    n = len(s)
    if n <= 32:
        return _len0to16(s, n) if n <= 16 else _len17to32(s, n)
    if n <= 64:
        return _len33to64(s, n)
    x = _f64(s, 0)
    y = _f64(s, n - 16) ^ K1
    z = _f64(s, n - 56) ^ K0
    v = _weak32s(s, n - 64, n, y)
    w = _weak32s(s, n - 32, (n * K1) & MASK, K0)
    z = (z + _shift_mix(v[1]) * K1) & MASK
    x = (_rot((z + x) & MASK, 39) * K1) & MASK
    y = (_rot(y, 33) * K1) & MASK
    remaining = (n - 1) & ~63
    i = 0
    while True:
        x = (_rot((x + y + v[0] + _f64(s, i + 16)) & MASK, 37) * K1) & MASK
        y = (_rot((y + v[1] + _f64(s, i + 48)) & MASK, 42) * K1) & MASK
        x ^= w[1]
        y ^= v[0]
        z = _rot(z ^ w[0], 33)
        v = _weak32s(s, i, (v[1] * K1) & MASK, (x + w[0]) & MASK)
        w = _weak32s(s, i + 32, (z + w[1]) & MASK, y)
        z, x = x, z
        i += 64
        remaining -= 64
        if remaining == 0:
            break
    return _hash128to64(
        (_hash128to64(v[0], w[0]) + _shift_mix(y) * K1 + z) & MASK,
        (_hash128to64(v[1], w[1]) + x) & MASK,
    )
