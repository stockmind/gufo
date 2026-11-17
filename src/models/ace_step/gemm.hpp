#ifndef GUFO_MODELS_ACE_STEP_GEMM_HPP_
#define GUFO_MODELS_ACE_STEP_GEMM_HPP_

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <memory>
#include <string>

namespace gufo::models::ace_step {

/// BF16 linear layers through hipBLASLt: row-major
/// out[tokens, rows] = bf16(in[tokens, columns] * weight[rows, columns]^T +
/// bias), FP32 accumulation, the bias added before the single rounding (as
/// torch's addmm epilogue). Each new shape is tuned once by timing the
/// library's heuristic candidates on the caller's buffers; launches are split
/// into chunks of at most kRowChunk tokens to stay short.
class LinearGemm {
public:
  static constexpr int kRowChunk = 4096;

  LinearGemm();
  ~LinearGemm();
  LinearGemm(const LinearGemm&) = delete;
  LinearGemm& operator=(const LinearGemm&) = delete;

  bool Init(hipStream_t stream, std::string* error);
  /// `bias` (BF16, length rows) may be null.
  bool Run(const std::uint16_t* weight, const std::uint16_t* bias,
           const std::uint16_t* input, std::uint16_t* out, int tokens, int rows,
           int columns, std::string* error);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::ace_step

#endif  // GUFO_MODELS_ACE_STEP_GEMM_HPP_
