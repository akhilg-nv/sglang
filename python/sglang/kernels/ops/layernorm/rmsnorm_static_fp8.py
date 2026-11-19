"""RMSNorm that also writes its output quantized to FP8 with a static per-tensor scale.

For a norm read both in BF16 and by an FP8 linear with a static input_scale
(ModelOptFp8LinearMethod): one launch writes the BF16 row and its FP8 codes,
quantized from the rounded BF16 row exactly as static_quant_fp8 of it would be.
The BF16 row follows sgl_kernel's rmsnorm math, but Triton's sum order may move
it by an ulp.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _rmsnorm_static_fp8_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    y_ptr,
    q_ptr,
    x_stride,
    N,
    eps,
    BLOCK_N: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    x = tl.load(x_ptr + row * x_stride + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y_s = tl.load(s_ptr).to(tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    rstd = tl.math.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    y = (x * rstd * w).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + row * N + offs, y, mask=mask)
    # static_quant_fp8's math on the rounded row: multiply by 1 / scale, saturate.
    y_s_inv = 1.0 / y_s
    y_q = tl.clamp(y.to(tl.float32) * y_s_inv, -448.0, 448.0)
    tl.store(q_ptr + row * N + offs, y_q.to(q_ptr.dtype.element_ty), mask=mask)


def _rmsnorm_static_fp8_fake(
    x: torch.Tensor, weight: torch.Tensor, eps: float, scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    shape = (x.shape[0], x.shape[1])
    return x.new_empty(shape), x.new_empty(shape, dtype=torch.float8_e4m3fn)


@register_custom_op(op_name="rmsnorm_static_fp8", fake_impl=_rmsnorm_static_fp8_fake)
def rmsnorm_static_fp8(
    x: torch.Tensor, weight: torch.Tensor, eps: float, scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm of the rows of ``x`` [M, N] (row stride free, unit column stride):
    ``(y, q)``, y [M, N] in x's dtype and q [M, N] float8_e4m3fn = y quantized
    with the per-tensor FP32 ``scale``."""
    assert x.dim() == 2 and x.stride(-1) == 1
    m, n = x.shape
    y = x.new_empty((m, n))
    q = x.new_empty((m, n), dtype=torch.float8_e4m3fn)
    if m == 0:
        return y, q
    block_n = triton.next_power_of_2(n)
    pdl_kwargs = {"USE_PDL": True, "launch_pdl": True} if is_arch_support_pdl() else {}
    _rmsnorm_static_fp8_kernel[(m,)](
        x,
        weight,
        scale,
        y,
        q,
        x.stride(0),
        n,
        eps,
        BLOCK_N=block_n,
        num_warps=min(max(block_n // 256, 1), 8),
        **pdl_kwargs,
    )
    return y, q
