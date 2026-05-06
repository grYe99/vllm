// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../../dispatch_utils.h"
#include "quant_conversions.cuh"
#include "../w8a8/fp8/common.cuh"

namespace vllm {

template <typename scalar_t, typename scalar_out_t, int NUM_THREADS>
__global__ void silu_and_mul_per_token_quant_kernel(
    scalar_out_t* __restrict__ out,      // Output: [num_tokens, hidden_size]
    float* __restrict__ scales,          // Output: [num_tokens]
    scalar_t const* __restrict__ input,  // Input: [num_tokens, hidden_size * 2]
    float const* scale_ub,               // Optional scale upper bound
    int32_t const hidden_size  // Output hidden size (input is 2x this)
) {
  // Grid: (num_tokens)
  int const token_idx = blockIdx.x;
  int const tid = threadIdx.x;
  int const stride = blockDim.x;

  // Input layout: [gate || up] concatenated along last dimension
  int const input_stride = hidden_size * 2;

  // Pointers to this token's data
  scalar_t const* token_input_gate = input + token_idx * input_stride;
  scalar_t const* token_input_up = token_input_gate + hidden_size;
  scalar_out_t* token_output = out + token_idx * hidden_size;

  float local_max = 0.0f;

  for (int i = tid; i < hidden_size; i += stride) {
    float gate = static_cast<float>(token_input_gate[i]);
    float up = static_cast<float>(token_input_up[i]);

    // Compute SiLU(gate) * up
    float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
    float silu_gate = gate * sigmoid_gate;
    float result = silu_gate * up;

    local_max = fmaxf(local_max, fabsf(result));
  }

  // Shared memory for reduction (compile-time sized)
  __shared__ float shared_max[NUM_THREADS];

  // Step 2: Reduce to find local max
  shared_max[tid] = local_max;
  __syncthreads();

// Power-of-2 reduction (NUM_THREADS guaranteed to be power of 2)
#pragma unroll
  for (int stride = NUM_THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
    }
    __syncthreads();
  }

  // Step 3: Compute scale (thread 0), broadcast via shared memory
  if (tid == 0) {
    float token_max = shared_max[0];
    float const quant_range = quant_type_max_v<scalar_out_t>;
    float token_scale = token_max / quant_range;

    // Apply scale upper bound if provided
    if (scale_ub != nullptr) {
      token_scale = fminf(token_scale, *scale_ub);
    }

    // Use minimum safe scaling factor
    token_scale = fmaxf(token_scale, min_scaling_factor<scalar_out_t>::val());

    // Store scale to global memory
    scales[token_idx] = token_scale;

    // Reuse shared_max[0] to broadcast scale
    shared_max[0] = token_scale;
  }
  __syncthreads();

  float token_scale = shared_max[0];

  // Step 4: Quantize and write output
  for (int i = tid; i < hidden_size; i += stride) {
    float gate = static_cast<float>(input[token_idx * (hidden_size * 2) + i]);
    float up = static_cast<float>(
        input[token_idx * (hidden_size * 2) + hidden_size + i]);

    float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
    float result = gate * sigmoid_gate * up;

    token_output[i] =
        vllm::ScaledQuant<scalar_out_t, false>::quant_fn(result, token_scale);
  }
}

}  // namespace vllm

void silu_and_mul_per_token_quant(torch::Tensor& out,
                                  torch::Tensor const& input,
                                  torch::Tensor& scales,
                                  std::optional<torch::Tensor> scale_ub) {
  static c10::ScalarType kFp8Type = is_fp8_ocp()
                                        ? c10::ScalarType::Float8_e4m3fn
                                        : c10::ScalarType::Float8_e4m3fnuz;

  TORCH_CHECK(out.dtype() == kFp8Type || out.dtype() == torch::kInt8);
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous());
  TORCH_CHECK(
      input.dtype() == torch::kFloat16 || input.dtype() == torch::kBFloat16,
      "Input must be FP16 or BF16");
  TORCH_CHECK(scales.dtype() == torch::kFloat32, "Scales must be FP32");

  if (scale_ub.has_value()) {
    TORCH_CHECK(out.dtype() == kFp8Type);
  }

  int32_t hidden_size = out.size(-1);
  auto num_tokens = input.size(0);

  TORCH_CHECK(input.size(-1) == hidden_size * 2,
              "input last dim must be 2x output hidden_size");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr int NUM_THREADS = 256;
  dim3 grid(num_tokens);
  dim3 block(NUM_THREADS);

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "silu_and_mul_per_token_quant", [&] {
        using scalar_in_t = scalar_t;

        VLLM_DISPATCH_QUANT_TYPES(
            out.scalar_type(), "silu_and_mul_per_token_quant", [&] {
              using scalar_out_t = scalar_t;
              vllm::silu_and_mul_per_token_quant_kernel<
                  scalar_in_t, scalar_out_t, NUM_THREADS>
                  <<<grid, block, 0, stream>>>(
                      out.data_ptr<scalar_out_t>(), scales.data_ptr<float>(),
                      input.data_ptr<scalar_in_t>(),
                      scale_ub.has_value() ? scale_ub->data_ptr<float>()
                                           : nullptr,
                      hidden_size);
            });
      });
}