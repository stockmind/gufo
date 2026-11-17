#ifndef GUFO_MODELS_ACE_STEP_ATTENTION_HIP_HPP_
#define GUFO_MODELS_ACE_STEP_ATTENTION_HIP_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::ace_step {

/// Transposes head-major BF16 values [heads][rows][128] into the layout the
/// attention kernel reads, FP16 [heads][128][PaddedRows(rows)] (clamped to
/// +-65504), zeroing the pad.
void LaunchPackValues(const std::uint16_t* values, std::uint16_t* packed,
                      std::uint32_t rows, std::uint32_t heads,
                      hipStream_t stream);

/// Rows rounded up to the 32-key tile of the attention kernel.
constexpr std::uint32_t PaddedRows(std::uint32_t rows) {
  return (rows + 31U) & ~31U;
}

/// Key visibility of one attention call. Query i sees key j when
/// (window == 0 || |i - j| <= window), (!causal || j <= i) and
/// (group == 0 || i / group == j / group); positions are row indices of the
/// query and key sequences. `capacity` > key_rows reads keys and packed
/// values from a cache of that many rows (key head stride capacity * 128,
/// packed values padded to PaddedRows(capacity)).
struct AttentionMask {
  std::uint32_t window{0};
  bool causal{false};
  std::uint32_t group{0};
  std::uint32_t capacity{0};
};

/// BF16 attention with head dimension 128 and scale 1/sqrt(128), grouped
/// queries: query head h reads key/value head h / (heads / kv_heads). Queries
/// [heads][query_rows][128] and keys [kv_heads][key_rows][128] are head-major,
/// values packed by LaunchPackValues (kv_heads). Writes row-major output
/// [query_rows][heads * 128]. Scores, softmax and the value accumulation are
/// FP32; probabilities and values enter the PV product as FP16. Launches are
/// split by query blocks to stay far below Windows' 2 s GPU watchdog.
void LaunchAttention(const std::uint16_t* query, const std::uint16_t* key,
                     const std::uint16_t* packed_values, std::uint16_t* output,
                     std::uint32_t query_rows, std::uint32_t key_rows,
                     std::uint32_t heads, std::uint32_t kv_heads,
                     AttentionMask mask, hipStream_t stream);

}  // namespace gufo::models::ace_step

#endif  // GUFO_MODELS_ACE_STEP_ATTENTION_HIP_HPP_
