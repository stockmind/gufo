#ifndef GUFO_MODELS_ACE_STEP_KERNELS_HPP_
#define GUFO_MODELS_ACE_STEP_KERNELS_HPP_

#include <hip/hip_runtime_api.h>

#include <cstdint>

namespace gufo::models::ace_step {

/// FP32 RMS norm of rows of `x` [rows, width] (eps 1e-6) times the BF16
/// `weight`, then, when `scale` / `shift` (FP32 [width]) are given,
/// y * (1 + scale) + shift; written as BF16 to `out` and/or FP32 to `out_f32`.
void LaunchRmsNorm(const float* x, const std::uint16_t* weight,
                   const float* scale, const float* shift, int rows, int width,
                   std::uint16_t* out, float* out_f32, hipStream_t stream);

/// Splits `heads` heads of 128 starting at column `column` of the BF16 rows
/// `linear` [rows, stride], RMS-normalizes each head with `norm_weight` [128]
/// (when not null), rotates it with RoPE (theta 1e6, rotate-half pairs
/// d / d + 64) when `rope`, and writes head-major BF16 [heads][rows][128].
/// The RoPE position is positions[row] (device ints) or the row, taken
/// modulo `period` when it is positive.
void LaunchHeads(const std::uint16_t* linear, int stride, int column, int rows,
                 int heads, const std::uint16_t* norm_weight, bool rope,
                 std::uint16_t* out, hipStream_t stream,
                 const int* positions = nullptr, int period = 0);

/// Writes head-major keys / values [kv_heads][rows][128] at rows
/// [start, start + rows) of a cache: keys [kv_heads][capacity][128] (BF16),
/// values packed for the attention kernel as FP16
/// [kv_heads][128][PaddedRows(capacity)].
void LaunchStoreCache(const std::uint16_t* keys, const std::uint16_t* values,
                      int rows, int kv_heads, int start, int capacity,
                      std::uint16_t* key_cache, std::uint16_t* value_cache,
                      hipStream_t stream);

/// out[r, f] = bf16(silu(x[r, f]) * x[r, features + f]) of BF16 x [rows,
/// 2 * features].
void LaunchSwiGlu(const std::uint16_t* x, int rows, int features,
                  std::uint16_t* out, hipStream_t stream);

/// hidden[r, c] += y[r, c] * (gate ? gate[c] : 1), y BF16, hidden FP32.
void LaunchResidual(float* hidden, const std::uint16_t* y, const float* gate,
                    int rows, int width, hipStream_t stream);

/// out[r, :] = FP32 table[ids[r], :] (BF16 table).
void LaunchGather(const std::uint16_t* table, const int* ids, int rows,
                  int width, float* out, hipStream_t stream);

void LaunchToBf16(const float* x, std::size_t count, std::uint16_t* out,
                  hipStream_t stream);
void LaunchToF32(const std::uint16_t* x, std::size_t count, float* out,
                 hipStream_t stream);

/// out[r] = bf16 round of (W[r, :] . act(in) + bias[r]), FP32 accumulation;
/// act is SiLU when `silu_input`. W and bias BF16; one input vector.
void LaunchMatVec(const std::uint16_t* weight, const std::uint16_t* bias,
                  const float* in, int columns, int rows, bool silu_input,
                  float* out, hipStream_t stream);

/// out[i] = BF16 table[i] + FP32 add[i] (count values).
void LaunchAddTable(const std::uint16_t* table, const float* add,
                    std::size_t count, float* out, hipStream_t stream);

/// out[n * parts + p, :] = in[n, :] + special[p, :] (FP32, width columns).
void LaunchExpand(const float* in, const float* special, int rows, int parts,
                  int width, float* out, hipStream_t stream);

/// x[i] += y[i] (FP32).
void LaunchAdd(float* x, const float* y, std::size_t count, hipStream_t stream);

/// Groups of `parts` rows of `in` [groups * parts, width] get `token`
/// [width] prepended: out[g * (parts + 1), :] = token, out[g * (parts + 1)
/// + 1 + p, :] = in[g * parts + p, :] (FP32; the attention pooler's
/// special token).
void LaunchGroupPrepend(const float* in, const float* token, int groups,
                        int parts, int width, float* out, hipStream_t stream);

}  // namespace gufo::models::ace_step

#endif  // GUFO_MODELS_ACE_STEP_KERNELS_HPP_
