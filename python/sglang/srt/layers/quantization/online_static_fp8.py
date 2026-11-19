"""Experimental: static per-tensor FP8 for linears a checkpoint leaves in BF16, with activation scales
calibrated by SGLang itself instead of ModelOpt.

Two steps, both selected per module prefix in modelopt_quant._bf16_linear_method:
  record  CalibratingLinearMethod keeps the BF16 GEMM and tracks max |input| per linear; every
          _DUMP_EVERY calls it writes {prefix: amax} to <SGLANG_MODELOPT_BF16_LINEAR_CALIB_DUMP>.tp<rank>.json.
          Merge the per-rank files with the max (kernel-benchmarking acc/merge_amax.py).
  apply   OnlineStaticFp8LinearMethod loads the BF16 weight, quantizes it per tensor at load time and uses
          input_scale = amax / 448 from SGLANG_MODELOPT_BF16_LINEAR_FP8_AMAX, i.e. ModelOpt's FP8 recipe
          (max calibration), so the layer runs exactly like a ModelOpt FP8 linear afterwards.
"""

import json
import logging
from functools import lru_cache
from types import SimpleNamespace
from typing import Dict, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

logger = logging.getLogger(__name__)

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
_DUMP_EVERY = 2000

_amax: Dict[str, torch.Tensor] = {}
_calls = 0


class CalibratingLinearMethod(UnquantizedLinearMethod):
    """BF16 linear that also records the running max |input| for its prefix (run with CUDA graphs off)."""

    def __init__(self, prefix: str, dump_path: str):
        super().__init__()
        self.prefix = prefix
        self.dump_path = dump_path

    def apply(self, layer, x, bias=None):
        global _calls
        if x.numel():
            m = x.detach().abs().amax().float()
            prev = _amax.get(self.prefix)
            _amax[self.prefix] = m if prev is None else torch.maximum(prev, m)
        _calls += 1
        if _calls % _DUMP_EVERY == 0:
            _dump(self.dump_path)
        return super().apply(layer, x, bias)


def _dump(path: str) -> None:
    from sglang.srt.runtime_context import get_parallel

    with open(f"{path}.tp{get_parallel().tp_rank}.json", "w") as f:
        json.dump({k: float(v) for k, v in _amax.items()}, f, indent=0)


class _StandInConfig(SimpleNamespace):
    """quant_config of the online methods. Model code asks a linear's config for get_name() (packed-weight checks) and
    weight_block_size (shared-expert and loader paths), so plain SimpleNamespace is not enough."""

    weight_block_size = None

    def get_name(self) -> str:
        return "modelopt_online"


@lru_cache(maxsize=None)
def load_amax(path: str) -> Dict[str, float]:
    with open(path) as f:
        return json.load(f)


def static_fp8_method(
    prefix: str, amax_path: str
) -> Optional["OnlineStaticFp8LinearMethod"]:
    amax = load_amax(amax_path).get(prefix)
    if not amax or amax <= 0:
        return None
    return OnlineStaticFp8LinearMethod(amax)


def _make_method_class():
    # Imported lazily: modelopt_quant imports this module's helpers.
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod

    class _OnlineStaticFp8LinearMethod(ModelOptFp8LinearMethod):
        """ModelOpt per-tensor FP8 linear built from a BF16 weight and a calibrated input amax."""

        def __init__(self, amax: float):
            super().__init__(_StandInConfig(is_checkpoint_fp8_serialized=False))
            self.amax = amax

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            w = layer.weight.data
            n_shards = len(layer.logical_widths)
            w_scale = (w.abs().amax().float() / FP8_MAX).clamp(min=1e-12)
            q = (w.float() / w_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            layer.weight = Parameter(q, requires_grad=False)
            layer.weight_scale = Parameter(
                w_scale.repeat(n_shards), requires_grad=False
            )
            layer.input_scale = Parameter(
                torch.full(
                    (n_shards,),
                    self.amax / FP8_MAX,
                    dtype=torch.float32,
                    device=w.device,
                ),
                requires_grad=False,
            )
            super().process_weights_after_loading(layer)

    return _OnlineStaticFp8LinearMethod


def OnlineStaticFp8LinearMethod(amax: float):  # noqa: N802 - factory standing in for the lazily built class
    global _cls
    try:
        cls = _cls
    except NameError:
        cls = _cls = _make_method_class()
    return cls(amax)


class CalibratingWrapper:
    """Wraps any linear method (e.g. block FP8) and records max |input| per prefix before delegating."""

    def __init__(self, inner, prefix: str, dump_path: str):
        self.inner, self.prefix, self.dump_path = inner, prefix, dump_path

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def create_weights(self, layer, *args, **kwargs):
        return self.inner.create_weights(layer, *args, **kwargs)

    def process_weights_after_loading(self, layer):
        return self.inner.process_weights_after_loading(layer)

    def apply(self, layer, x, bias=None):
        global _calls
        if isinstance(x, torch.Tensor) and x.numel():
            m = x.detach().abs().amax().float()
            prev = _amax.get(self.prefix)
            _amax[self.prefix] = m if prev is None else torch.maximum(prev, m)
            _calls += 1
            if _calls % _DUMP_EVERY == 0:
                _dump(self.dump_path)
        return self.inner.apply(layer, x, bias)


def block_fp8_as_tensor_method(block_method, prefix: str, amax_path: str):
    """Static per-tensor FP8 for a block-FP8 checkpoint linear: loads the block weights and scales as the block
    method would, then dequantizes and requantizes per tensor; input_scale = calibrated amax / 448."""
    amax = load_amax(amax_path).get(prefix)
    if not amax or amax <= 0:
        return None
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod

    class _BlockFp8AsTensorFp8(ModelOptFp8LinearMethod):
        def __init__(self):
            # weight_block_size: the v2 weight loader reads it to shard the block scales
            super().__init__(
                _StandInConfig(
                    is_checkpoint_fp8_serialized=True,
                    weight_block_size=block_method.quant_config.weight_block_size,
                )
            )
            # input_scale = amax / 448: linears with equal amax get the same scale (model code may share an input).
            self.amax = amax

        def create_weights(self, layer, *args, **kwargs):
            block_method.create_weights(layer, *args, **kwargs)

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            bn, bk = block_method.quant_config.weight_block_size
            w, s = layer.weight.data, layer.weight_scale_inv.data.float()
            n, k = w.shape
            s = s.repeat_interleave(bn, 0)[:n].repeat_interleave(bk, 1)[:, :k]
            w = w.float() * s
            w_scale = (w.abs().amax() / FP8_MAX).clamp(min=1e-12)
            q = (w / w_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            del w, s
            del layer.weight_scale_inv
            n_shards = len(layer.logical_widths)
            layer.weight = Parameter(q, requires_grad=False)
            layer.weight_scale = Parameter(
                w_scale.repeat(n_shards), requires_grad=False
            )
            layer.input_scale = Parameter(
                torch.full(
                    (n_shards,), amax / FP8_MAX, dtype=torch.float32, device=q.device
                ),
                requires_grad=False,
            )
            super().process_weights_after_loading(layer)

    return _BlockFp8AsTensorFp8()


FP4_MAX = 6.0


def static_nvfp4_method(prefix: str, amax_path: str):
    """NVFP4 W4A4 (ModelOpt layout) for a BF16 linear: weights quantized at load (global scale from the weight max),
    input global scale from the calibrated amax, i.e. ModelOpt's NVFP4 recipe with max calibration."""
    amax = load_amax(amax_path).get(prefix)
    if not amax or amax <= 0:
        return None
    from sglang.srt.layers.parameter import ModelWeightParameter
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4LinearMethod

    config = _StandInConfig(
        is_checkpoint_nvfp4_serialized=False, group_size=16, is_awq=False
    )

    class _OnlineNvfp4LinearMethod(ModelOptFp4LinearMethod):
        def create_weights(
            self,
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        ):
            output_size_per_partition = sum(output_partition_sizes)
            layer.logical_widths = output_partition_sizes
            layer.input_size_per_partition = input_size_per_partition
            layer.output_size_per_partition = output_size_per_partition
            layer.params_dtype = params_dtype
            layer.quant_config = config
            layer.register_parameter(
                "weight",
                ModelWeightParameter(
                    data=torch.empty(
                        output_size_per_partition,
                        input_size_per_partition,
                        dtype=params_dtype,
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=extra_weight_attrs.get("weight_loader"),
                ),
            )

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            from flashinfer import fp4_quantize

            w = layer.weight.data
            n, k = w.shape
            n_shards = len(layer.logical_widths)
            global_w = (
                FP8_MAX * FP4_MAX / w.abs().amax().float().clamp(min=1e-12)
            ).reshape(1)
            q, sf = fp4_quantize(
                w.contiguous(),
                global_w,
                sf_vec_size=16,
                sf_use_ue8m0=False,
                is_sf_swizzled_layout=False,
            )
            layer.weight = Parameter(
                q.view(torch.uint8).reshape(n, k // 2), requires_grad=False
            )
            layer.weight_scale = Parameter(
                sf.view(torch.float8_e4m3fn).reshape(n, k // 16), requires_grad=False
            )
            layer.weight_scale_2 = Parameter(
                (1.0 / global_w).repeat(n_shards), requires_grad=False
            )
            layer.input_scale = Parameter(
                torch.full(
                    (n_shards,),
                    amax / (FP8_MAX * FP4_MAX),
                    dtype=torch.float32,
                    device=w.device,
                ),
                requires_grad=False,
            )
            super().process_weights_after_loading(layer)

    return _OnlineNvfp4LinearMethod(config)
