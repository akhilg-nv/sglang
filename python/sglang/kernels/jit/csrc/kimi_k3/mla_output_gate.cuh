// K3 MLA output gate: out = bf16(bf16(x) * bf16(sigmoid(gate))), replacing
// the torch.sigmoid + mul elementwise pair (two launches, two memory passes)
// with one kernel. sigmoid is computed in fp32 and rounded to bf16 before the
// multiply, reproducing the unfused pair's double rounding bit-for-bit.
// run_static_fp8 writes that bf16 result quantized to FP8 with a static
// per-tensor scale instead, byte for byte as static_quant_fp8 of it would.

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/math.cuh>   // For device::math::min, max
#include <sgl_kernel/type.cuh>   // For bf16_t, fp32_t, fp8_e4m3_t, device::cast
#include <sgl_kernel/utils.cuh>  // For LaunchKernel
#include <sgl_kernel/vec.cuh>    // For AlignedVector

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

struct MlaOutputGateParams {
  const bf16_t* __restrict__ x;      // [N] contiguous (flattened [T, H])
  const bf16_t* __restrict__ gate;   // [N] contiguous
  bf16_t* __restrict__ out;          // [N] contiguous; unused with kStaticFp8
  fp8_e4m3_t* __restrict__ out_fp8;  // [N] contiguous; kStaticFp8 only
  const float* __restrict__ scale;   // [1] static per-tensor FP8 scale; kStaticFp8 only
  uint32_t n_vecs;
};

SGL_DEVICE float triton_scale_reciprocal(float scale) {
  float reciprocal;
  // Triton's static FP8 quantizer lowers `1.0 / scale` to div.full.f32.
  // Match it exactly because a one-ULP difference at an E4M3 midpoint can
  // change the quantized byte.
  asm("div.full.f32 %0, %1, %2;" : "=f"(reciprocal) : "f"(1.0f), "f"(scale));
  return reciprocal;
}

template <int kThreads, bool kUsePDL, bool kStaticFp8>
__global__ void mla_output_gate_kernel(const MlaOutputGateParams __grid_constant__ params) {
  using namespace device;

  constexpr int kVecN = 8;
  using vec_bf16_t = AlignedVector<bf16_t, kVecN>;

  const uint32_t v = blockIdx.x * kThreads + threadIdx.x;
  if (v >= params.n_vecs) return;

  PDLWaitPrimary<kUsePDL>();

  vec_bf16_t xv, gv, ov;
  xv.load(params.x, v);
  gv.load(params.gate, v);
#pragma unroll
  for (int i = 0; i < kVecN; ++i) {
    // Match torch.sigmoid(bf16): fp32 sigmoid, round to bf16, then the bf16
    // multiply upcasts both operands to fp32 and rounds once more.
    const float g = cast<fp32_t>(gv[i]);
    const bf16_t s = cast<bf16_t>(1.0f / (1.0f + expf(-g)));
    ov[i] = cast<bf16_t>(cast<fp32_t>(xv[i]) * cast<fp32_t>(s));
  }
  if constexpr (kStaticFp8) {
    // static_quant_fp8's math on the rounded product: multiply by 1 / scale, saturate.
    const float scale_inv = triton_scale_reciprocal(*params.scale);
    AlignedVector<fp8_e4m3_t, kVecN> qv;
#pragma unroll
    for (int i = 0; i < kVecN; ++i) {
      const float scaled = cast<fp32_t>(ov[i]) * scale_inv;
      qv[i] = static_cast<fp8_e4m3_t>(
          math::min(math::max(scaled, -DTypeTrait<fp8_e4m3_t>::kFloatMax), DTypeTrait<fp8_e4m3_t>::kFloatMax));
    }
    qv.store(params.out_fp8, v);
  } else {
    ov.store(params.out, v);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int kThreads, bool kUsePDL>
struct MlaOutputGateKernel {
  static constexpr auto kernel = mla_output_gate_kernel<kThreads, kUsePDL, false>;
  static constexpr auto kernel_static_fp8 = mla_output_gate_kernel<kThreads, kUsePDL, true>;

  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView gate, const tvm::ffi::TensorView out) {
    using namespace host;

    auto N_ = SymbolicSize{"numel"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N_}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N_}).with_dtype<bf16_t>().with_device(device).verify(gate);
    TensorMatcher({N_}).with_dtype<bf16_t>().with_device(device).verify(out);

    const auto N = static_cast<uint32_t>(N_.unwrap());
    RuntimeCheck(N % 8 == 0, "numel must be divisible by 8");
    if (N == 0) return;

    const auto params = MlaOutputGateParams{
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .gate = static_cast<const bf16_t*>(gate.data_ptr()),
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .out_fp8 = nullptr,
        .scale = nullptr,
        .n_vecs = N / 8,
    };
    const uint32_t n_blocks = (params.n_vecs + kThreads - 1) / kThreads;
    LaunchKernel(n_blocks, kThreads, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }

  static void run_static_fp8(
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView gate,
      const tvm::ffi::TensorView scale,
      const tvm::ffi::TensorView out_fp8) {
    using namespace host;

    auto N_ = SymbolicSize{"numel"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N_}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N_}).with_dtype<bf16_t>().with_device(device).verify(gate);
    TensorMatcher({1}).with_dtype<fp32_t>().with_device(device).verify(scale);
    TensorMatcher({N_}).with_dtype<fp8_e4m3_t>().with_device(device).verify(out_fp8);

    const auto N = static_cast<uint32_t>(N_.unwrap());
    RuntimeCheck(N % 8 == 0, "numel must be divisible by 8");
    if (N == 0) return;

    const auto params = MlaOutputGateParams{
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .gate = static_cast<const bf16_t*>(gate.data_ptr()),
        .out = nullptr,
        .out_fp8 = static_cast<fp8_e4m3_t*>(out_fp8.data_ptr()),
        .scale = static_cast<const float*>(scale.data_ptr()),
        .n_vecs = N / 8,
    };
    const uint32_t n_blocks = (params.n_vecs + kThreads - 1) / kThreads;
    LaunchKernel(n_blocks, kThreads, device.unwrap()).enable_pdl(kUsePDL)(kernel_static_fp8, params);
  }
};

}  // namespace sglang
