"""MXFP8 quantization written as an epilogue of the kernel that produces the row.

At the speculative BS=1 shapes (<=8 rows) the standalone FlashInfer
``mxfp8_quantize`` launch costs about as much as the norm it follows, even
though it only re-reads what that norm just wrote.  These kernels do the norm
and the quantization in one pass; the quantized values come from the same BF16
rounding the standalone pair produces, so both outputs are bitwise identical to
``rmsnorm`` followed by ``mxfp8_quantize(..., is_sf_swizzled_layout=True)``.

The scale factors use FlashInfer's 128x4 swizzle:
``off = ((r // 128) * ceil(G / 4) + g // 4) * 512
+ ((r % 32) * 4 + ((r // 32) % 4)) * 4 + (g % 4)`` for G scale columns,
over a row count padded to a multiple of 128, with the UE8M0 conversion
(positive rounding, subnormals included) that ``mxfp8_quantize`` uses.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# FlashInfer's positive-rounding UE8M0 conversion of a per-group amax, subnormals
# included: the scale byte and the multiplier that maps the group into e4m3 range.
@triton.jit
def ue8m0_scale(amax):
    normalized = amax * (1.0 / 448.0)
    bits = normalized.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
    sf = tl.where(normalized <= 0, 0, tl.minimum(exponent + bump.to(tl.int32), 254))
    inv = tl.where(sf == 0, 0, ((254 - sf) << 23)).to(tl.float32, bitcast=True)
    return sf, inv


@triton.jit
def sf_offset(row, g, GROUPS: tl.constexpr):
    # Byte offset of scale (row, g) in FlashInfer's 128x4 layout.
    tile = (row // 128) * ((GROUPS + 3) // 4) + g // 4
    return tile * 512 + ((row % 32) * 4 + ((row // 32) % 4)) * 4 + (g % 4)


@triton.jit
def mxfp8_epilogue(
    y, row, Q, S, K: tl.constexpr, BLOCK: tl.constexpr, GROUPS: tl.constexpr, g_lo, g_hi
):
    # Stores only groups in [g_lo, g_hi) so a row can be split across CTAs.
    GP: tl.constexpr = BLOCK // 32
    g = tl.arange(0, GP)
    gmask = (g < GROUPS) & (g >= g_lo) & (g < g_hi)
    e = tl.arange(0, 32)
    idx = g[:, None] * 32 + e[None, :]
    v = tl.reshape(y.to(tl.float32), (GP, 32))
    amax = tl.max(tl.abs(v), 1)
    sf, scale = ue8m0_scale(amax)
    q = tl.minimum(tl.maximum(v * scale[:, None], -448.0), 448.0).to(tl.float8e4nv)
    tl.store(Q + row * K + idx, q, gmask[:, None])
    tl.store(S + sf_offset(row, g, GROUPS), sf.to(tl.uint8), gmask)


@triton.jit
def _rmsnorm_mxfp8_kernel(
    X,
    W,
    Y,
    Q,
    S,
    SX: tl.constexpr,
    M,
    K: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPS: tl.constexpr,
):
    row = tl.program_id(0)
    if row < M:
        h = tl.arange(0, BLOCK)
        v = tl.load(X + row * SX + h, h < K, 0).to(tl.float32)
        weight = tl.load(W + h, h < K, 0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(v * v, 0) / K + EPS)
        y = (v * inv * weight).to(tl.bfloat16)
        tl.store(Y + row * K + h, y, h < K)
        mxfp8_epilogue(y, row, Q, S, K, BLOCK, GROUPS, 0, GROUPS)
    else:
        # Padding scale entries are disjoint from the live rows above.
        off = (row - M) * 512 + tl.arange(0, 512)
        pad_row = ((off // 4) % 4) * 32 + ((off // 16) % 32)
        tl.store(S + off, 0, (off < GROUPS * 128) & (pad_row >= M))


def rmsnorm_mxfp8(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMSNorm returning ``(y_bf16, y_q, y_sf)`` with the same MXFP8 epilogue."""
    m, k = x.shape
    # The padding below addresses a single 128-row scale tile.
    assert 0 < m <= 128 and k % 32 == 0
    assert x.dtype == weight.dtype == torch.bfloat16 and x.stride(1) == 1
    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((k // 32) * 128, dtype=torch.uint8, device=x.device)
    _rmsnorm_mxfp8_kernel[(m + triton.cdiv(s.numel(), 512),)](
        x,
        weight,
        y,
        q,
        s,
        SX=x.stride(0),
        M=m,
        K=k,
        EPS=eps,
        BLOCK=triton.next_power_of_2(k),
        GROUPS=k // 32,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return y, q, s


@triton.jit
def _add_rmsnorm_mxfp8_kernel(
    X,
    R,
    W,
    Y,
    Q,
    S,
    SX,
    SR,
    M,
    K: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_Y: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.arange(0, BLOCK)
    if row < M:
        # As fused_add_rmsnorm: the residual keeps the BF16 sum, the norm reads it in FP32.
        v = tl.load(X + row * SX + h, h < K, 0).to(tl.float32)
        if HAS_RESIDUAL:
            v += tl.load(R + row * SR + h, h < K, 0).to(tl.float32)
            tl.store(R + row * SR + h, v.to(tl.bfloat16), h < K)
        weight = tl.load(W + h, h < K, 0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(v * v, 0) / K + EPS)
        y = (v * inv * weight).to(tl.bfloat16)
        if STORE_Y:
            tl.store(Y + row * K + h, y, h < K)
        mxfp8_epilogue(y, row, Q, S, K, BLOCK, GROUPS, 0, GROUPS)
    else:
        # A padding row of the last 128-row tile: zero scales, as mxfp8_quantize writes.
        g = tl.arange(0, BLOCK // 32)
        tl.store(
            S + sf_offset(row, g, GROUPS), tl.zeros_like(g).to(tl.uint8), g < GROUPS
        )


def add_rmsnorm_mxfp8(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float,
    *,
    store_y: bool,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """``fused_add_rmsnorm`` (``residual += x`` in place, then RMSNorm of the sum), or
    ``rmsnorm`` without a residual, with the MXFP8 epilogue on its BF16 result, for any
    row count. Returns ``(y_bf16 if store_y else None, y_q, y_sf)``; ``y_sf`` is the
    128x4-swizzled scale buffer ``mxfp8_quantize`` returns, padding rows zero."""
    m, k = x.shape
    assert m > 0 and k % 128 == 0, "scale columns must fill whole 4-column tiles"
    assert x.dtype == weight.dtype == torch.bfloat16 and x.stride(1) == 1
    if residual is not None:
        assert residual.shape == x.shape and residual.dtype == x.dtype
        assert residual.stride(1) == 1
    padded_m = triton.cdiv(m, 128) * 128
    y = x.new_empty((m, k)) if store_y else None
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(padded_m * (k // 32), dtype=torch.uint8, device=x.device)
    _add_rmsnorm_mxfp8_kernel[(padded_m,)](
        x,
        residual,
        weight,
        y,
        q,
        s,
        x.stride(0),
        residual.stride(0) if residual is not None else 0,
        m,
        K=k,
        EPS=eps,
        BLOCK=triton.next_power_of_2(k),
        GROUPS=k // 32,
        HAS_RESIDUAL=residual is not None,
        STORE_Y=store_y,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return y, q, s
