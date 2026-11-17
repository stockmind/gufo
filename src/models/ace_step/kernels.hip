#include <hip/hip_runtime.h>

#include <cmath>

#include "src/models/ace_step/kernels.hpp"

namespace gufo::models::ace_step {
namespace {

constexpr int kThreads = 256;
constexpr float kEpsilon = 1e-6F;

__device__ __forceinline__ float FromBf16(std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t ToBf16(float value) {
  const unsigned int bits = __float_as_uint(value);
  if ((bits & 0x7FFFFFFFU) > 0x7F800000U)
    return static_cast<std::uint16_t>((bits >> 16U) | 0x40U);
  return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16U) & 1U)) >>
                                    16U);
}

__device__ __forceinline__ float Silu(float x) {
  return x / (1.0F + expf(-x));
}

template<int kCount>
__device__ __forceinline__ float BlockSum(float value, float* shared) {
  const int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();
#pragma unroll
  for (int stride = kCount / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      shared[tid] += shared[tid + stride];
    __syncthreads();
  }
  const float result = shared[0];
  __syncthreads();
  return result;
}

__global__ void RmsNormKernel(const float* x, const std::uint16_t* weight,
                              const float* scale, const float* shift, int width,
                              std::uint16_t* out, float* out_f32) {
  __shared__ float shared[kThreads];
  const std::size_t row = blockIdx.x;
  const float* in = x + row * width;
  float squares = 0;
  for (int c = threadIdx.x; c < width; c += kThreads)
    squares += in[c] * in[c];
  const float rstd =
      rsqrtf(BlockSum<kThreads>(squares, shared) / width + kEpsilon);
  for (int c = threadIdx.x; c < width; c += kThreads) {
    float y = in[c] * rstd * FromBf16(weight[c]);
    if (scale != nullptr)
      y = y * (1.0F + scale[c]) + shift[c];
    if (out != nullptr)
      out[row * width + c] = ToBf16(y);
    if (out_f32 != nullptr)
      out_f32[row * width + c] = y;
  }
}

// One block per (row, head), one thread per head dimension.
__global__ void HeadsKernel(const std::uint16_t* linear, int stride, int column,
                            int rows, const std::uint16_t* norm_weight,
                            bool rope, std::uint16_t* out, const int* positions,
                            int period) {
  __shared__ float shared[128];
  __shared__ float values[128];
  const int row = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
  float v = FromBf16(
      linear[static_cast<std::size_t>(row) * stride + column + head * 128 + d]);
  if (norm_weight != nullptr) {
    const float rstd = rsqrtf(BlockSum<128>(v * v, shared) / 128.0F + kEpsilon);
    v = v * rstd * FromBf16(norm_weight[d]);
  }
  if (rope) {
    values[d] = v;
    __syncthreads();
    const int pair = d % 64;
    // inv_freq = 1 / 1e6^(2 * pair / 128), FP32 as torch builds it.
    const float inv =
        1.0F / powf(1000000.0F, static_cast<float>(2 * pair) / 128.0F);
    int position = positions != nullptr ? positions[row] : row;
    if (period > 0)
      position %= period;
    const float angle = static_cast<float>(position) * inv;
    const float c = cosf(angle), s = sinf(angle);
    v = d < 64 ? v * c - values[d + 64] * s : v * c + values[d - 64] * s;
  }
  out[(static_cast<std::size_t>(head) * rows + row) * 128 + d] = ToBf16(v);
}

__global__ void SwiGluKernel(const std::uint16_t* x, int rows, int features,
                             std::uint16_t* out) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::size_t>(rows) * features)
    return;
  const std::size_t r = index / features, f = index % features;
  const float g = FromBf16(x[r * 2 * features + f]);
  const float u = FromBf16(x[r * 2 * features + features + f]);
  out[index] = ToBf16(Silu(g) * u);
}

__global__ void ResidualKernel(float* hidden, const std::uint16_t* y,
                               const float* gate, std::size_t count,
                               int width) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const float v = FromBf16(y[i]);
  hidden[i] += gate != nullptr ? v * gate[i % width] : v;
}

__global__ void GatherKernel(const std::uint16_t* table, const int* ids,
                             int width, float* out) {
  const std::size_t row = blockIdx.x;
  const std::uint16_t* source =
      table + static_cast<std::size_t>(ids[row]) * width;
  for (int c = threadIdx.x; c < width; c += kThreads)
    out[row * width + c] = FromBf16(source[c]);
}

__global__ void ToBf16Kernel(const float* x, std::size_t count,
                             std::uint16_t* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = ToBf16(x[i]);
}

__global__ void ToF32Kernel(const std::uint16_t* x, std::size_t count,
                            float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = FromBf16(x[i]);
}

__global__ void MatVecKernel(const std::uint16_t* weight,
                             const std::uint16_t* bias, const float* in,
                             int columns, bool silu_input, float* out) {
  __shared__ float shared[kThreads];
  const int r = blockIdx.x;
  const std::uint16_t* w = weight + static_cast<std::size_t>(r) * columns;
  float sum = 0;
  for (int c = threadIdx.x; c < columns; c += kThreads) {
    const float v = silu_input ? Silu(in[c]) : in[c];
    sum += FromBf16(w[c]) * v;
  }
  sum = BlockSum<kThreads>(sum, shared);
  if (threadIdx.x == 0)
    out[r] = sum + (bias != nullptr ? FromBf16(bias[r]) : 0.0F);
}

__global__ void AddTableKernel(const std::uint16_t* table, const float* add,
                               std::size_t count, float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = FromBf16(table[i]) + add[i];
}

__global__ void ExpandKernel(const float* in, const float* special, int parts,
                             int width, std::size_t count, float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::size_t c = i % width, row = i / width;
  const std::size_t n = row / parts, p = row % parts;
  out[i] = in[n * width + c] + special[p * width + c];
}

__global__ void GroupPrependKernel(const float* in, const float* token,
                                   int parts, int width, std::size_t count,
                                   float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const std::size_t c = i % width, row = i / width;
  const std::size_t g = row / (parts + 1), p = row % (parts + 1);
  out[i] = p == 0 ? token[c] : in[(g * parts + p - 1) * width + c];
}

__global__ void AddKernel(float* x, const float* y, std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    x[i] += y[i];
}

__global__ void StoreCacheKernel(const std::uint16_t* keys,
                                 const std::uint16_t* values, int rows,
                                 int start, int capacity, std::size_t count,
                                 std::uint16_t* key_cache,
                                 std::uint16_t* value_cache) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const int d = static_cast<int>(i % 128);
  const int r = static_cast<int>((i / 128) % rows);
  const std::size_t h = i / (128 * static_cast<std::size_t>(rows));
  const int position = start + r;
  key_cache[(h * capacity + position) * 128 + d] = keys[i];
  const int padded = (capacity + 31) & ~31;
  const float v = fminf(fmaxf(FromBf16(values[i]), -65504.0F), 65504.0F);
  value_cache[(h * 128 + d) * padded + position] =
      __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(v));
}

unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

}  // namespace

void LaunchRmsNorm(const float* x, const std::uint16_t* weight,
                   const float* scale, const float* shift, int rows, int width,
                   std::uint16_t* out, float* out_f32, hipStream_t stream) {
  RmsNormKernel<<<rows, kThreads, 0, stream>>>(x, weight, scale, shift, width,
                                               out, out_f32);
}

void LaunchHeads(const std::uint16_t* linear, int stride, int column, int rows,
                 int heads, const std::uint16_t* norm_weight, bool rope,
                 std::uint16_t* out, hipStream_t stream, const int* positions,
                 int period) {
  HeadsKernel<<<dim3(rows, heads), 128, 0, stream>>>(
      linear, stride, column, rows, norm_weight, rope, out, positions, period);
}

void LaunchStoreCache(const std::uint16_t* keys, const std::uint16_t* values,
                      int rows, int kv_heads, int start, int capacity,
                      std::uint16_t* key_cache, std::uint16_t* value_cache,
                      hipStream_t stream) {
  const std::size_t count = std::size_t(kv_heads) * rows * 128;
  StoreCacheKernel<<<Blocks(count), kThreads, 0, stream>>>(
      keys, values, rows, start, capacity, count, key_cache, value_cache);
}

void LaunchSwiGlu(const std::uint16_t* x, int rows, int features,
                  std::uint16_t* out, hipStream_t stream) {
  const std::size_t count = static_cast<std::size_t>(rows) * features;
  SwiGluKernel<<<Blocks(count), kThreads, 0, stream>>>(x, rows, features, out);
}

void LaunchResidual(float* hidden, const std::uint16_t* y, const float* gate,
                    int rows, int width, hipStream_t stream) {
  const std::size_t count = static_cast<std::size_t>(rows) * width;
  ResidualKernel<<<Blocks(count), kThreads, 0, stream>>>(hidden, y, gate, count,
                                                         width);
}

void LaunchGather(const std::uint16_t* table, const int* ids, int rows,
                  int width, float* out, hipStream_t stream) {
  GatherKernel<<<rows, kThreads, 0, stream>>>(table, ids, width, out);
}

void LaunchToBf16(const float* x, std::size_t count, std::uint16_t* out,
                  hipStream_t stream) {
  ToBf16Kernel<<<Blocks(count), kThreads, 0, stream>>>(x, count, out);
}

void LaunchToF32(const std::uint16_t* x, std::size_t count, float* out,
                 hipStream_t stream) {
  ToF32Kernel<<<Blocks(count), kThreads, 0, stream>>>(x, count, out);
}

void LaunchMatVec(const std::uint16_t* weight, const std::uint16_t* bias,
                  const float* in, int columns, int rows, bool silu_input,
                  float* out, hipStream_t stream) {
  MatVecKernel<<<rows, kThreads, 0, stream>>>(weight, bias, in, columns,
                                              silu_input, out);
}

void LaunchAddTable(const std::uint16_t* table, const float* add,
                    std::size_t count, float* out, hipStream_t stream) {
  AddTableKernel<<<Blocks(count), kThreads, 0, stream>>>(table, add, count,
                                                         out);
}

void LaunchExpand(const float* in, const float* special, int rows, int parts,
                  int width, float* out, hipStream_t stream) {
  const std::size_t count = std::size_t(rows) * parts * width;
  ExpandKernel<<<Blocks(count), kThreads, 0, stream>>>(in, special, parts,
                                                       width, count, out);
}

void LaunchAdd(float* x, const float* y, std::size_t count,
               hipStream_t stream) {
  AddKernel<<<Blocks(count), kThreads, 0, stream>>>(x, y, count);
}

void LaunchGroupPrepend(const float* in, const float* token, int groups,
                        int parts, int width, float* out, hipStream_t stream) {
  const std::size_t count = std::size_t(groups) * (parts + 1) * width;
  GroupPrependKernel<<<Blocks(count), kThreads, 0, stream>>>(
      in, token, parts, width, count, out);
}

}  // namespace gufo::models::ace_step
