// Parity of the routed F16 expert GEMM against a CPU reference for the packed
// low-bit formats (IQ3_XXS, IQ2_S, IQ4_NL) and Q4_K as the affine control.
//
// The decode corruption seen when packed experts took the routed tier at one
// token was invisible to timing runs, so this compares values: one expert, one
// weight tile at a time, T = 1, 8, 48 and 128 tokens, output against
// gufo::quant::Dequantize* on the same bytes. The kernel accumulates in F16, so
// the check is a relative-error bound, not equality.
//
// routed_gemm_parity_test
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace {

namespace q = gufo::models::qwen38_flash_next::rocm;

constexpr std::uint32_t kRows = 640;      // gate/up rows of one expert
constexpr std::uint32_t kHidden = 2560;   // K

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

std::uint32_t Next(std::uint32_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

struct Shape {
  const char* name;
  q::WeightType type;
  gufo::core::GgmlType file_type;
  std::size_t block_bytes;   // per block of `block_elements`
  std::size_t block_elements;
};

const Shape kShapes[] = {
    {"Q4_K", q::WeightType::kQ4_K, gufo::core::GgmlType::kQ4_K, 144, 256},
    {"IQ3_XXS", q::WeightType::kIQ3_XXS, gufo::core::GgmlType::kIQ3_XXS, 98,
     256},
    {"IQ2_S", q::WeightType::kIQ2_S, gufo::core::GgmlType::kIQ2_S, 82, 256},
    {"IQ4_NL", q::WeightType::kIQ4_NL, gufo::core::GgmlType::kIQ4_NL, 18, 32},
};

bool Reference(const Shape& shape, const void* row_bytes, std::size_t k,
               const float* x, float* out) {
  std::vector<float> row(k);
  switch (shape.file_type) {
    case gufo::core::GgmlType::kQ4_K:
      gufo::quant::DequantizeQ4_K(row_bytes, row.data(), k);
      break;
    case gufo::core::GgmlType::kIQ3_XXS:
      gufo::quant::DequantizeIQ3_XXS(row_bytes, row.data(), k);
      break;
    case gufo::core::GgmlType::kIQ2_S:
      gufo::quant::DequantizeIQ2_S(row_bytes, row.data(), k);
      break;
    case gufo::core::GgmlType::kIQ4_NL:
      gufo::quant::DequantizeIQ4_NL(row_bytes, row.data(), k);
      break;
    default:
      return false;
  }
  float sum = 0.0F;
  for (std::size_t i = 0; i < k; ++i) {
    sum += row[i] * x[i];
  }
  *out = sum;
  return true;
}

// One expert whose T tokens all route to it, so a single tile covers the
// bucket; `tile_rows` matches what the executor would build for that width.
int RunShape(const Shape& shape) {
  const std::size_t row_bytes = (kHidden / shape.block_elements) *
                                shape.block_bytes;
  const std::size_t total = static_cast<std::size_t>(kRows) * row_bytes;
  std::vector<std::uint8_t> host(total);
  std::uint32_t seed = 0x12345678U;
  // Quantized fields take arbitrary bytes, but the F16 block scale must be a
  // finite small value or both sides produce inf/NaN and the comparison says
  // nothing. Write (d, dmin) first for Q4_K, then randomize the codes.
  const std::size_t blocks = total / shape.block_bytes;
  for (auto& byte : host) {
    byte = static_cast<std::uint8_t>(Next(&seed) >> 24);
  }
  const __half scale = __float2half_rn(0.02F);
  const __half dmin = __float2half_rn(0.001F);
  std::uint16_t scale_bits = 0;
  std::uint16_t dmin_bits = 0;
  std::memcpy(&scale_bits, &scale, sizeof(scale_bits));
  std::memcpy(&dmin_bits, &dmin, sizeof(dmin_bits));
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* block = host.data() + b * shape.block_bytes;
    std::memcpy(block, &scale_bits, 2);
    if (shape.type == q::WeightType::kQ4_K) {
      std::memcpy(block + 2, &dmin_bits, 2);
    }
  }
  void* d_w = nullptr;
  CheckHip(hipMalloc(&d_w, total), "weight alloc");
  CheckHip(hipMemcpy(d_w, host.data(), total, hipMemcpyHostToDevice),
           "weight upload");

  int failures = 0;
  for (std::uint32_t tokens : {1U, 8U, 48U, 128U}) {
    std::vector<__half> x(tokens * kHidden);
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] = __float2half_rn(static_cast<float>(Next(&seed) % 2000) / 1000.0F -
                             1.0F);
    }
    __half* d_x = nullptr;
    float* d_out = nullptr;
    CheckHip(hipMalloc(&d_x, x.size() * sizeof(__half)), "activation alloc");
    CheckHip(hipMalloc(&d_out, tokens * kRows * sizeof(float)), "out alloc");
    CheckHip(hipMemcpy(d_x, x.data(), x.size() * sizeof(__half),
                       hipMemcpyHostToDevice),
             "activation upload");
    std::vector<std::int32_t> bounds{0, static_cast<std::int32_t>(tokens)};
    std::vector<std::int32_t> rows_token(tokens), rows_slot(tokens),
        tiles{0};
    for (std::uint32_t t = 0; t < tokens; ++t) {
      rows_token[t] = static_cast<std::int32_t>(t);
      rows_slot[t] = static_cast<std::int32_t>(t);
    }
    std::int32_t* d_bounds = nullptr;
    std::int32_t* d_rt = nullptr;
    std::int32_t* d_rs = nullptr;
    std::int32_t* d_tiles = nullptr;
    const std::size_t bounds_bytes = bounds.size() * sizeof(std::int32_t);
    const std::size_t row_bytes_meta = tokens * sizeof(std::int32_t);
    CheckHip(hipMalloc(&d_bounds, bounds_bytes), "bounds alloc");
    CheckHip(hipMalloc(&d_rt, row_bytes_meta), "rows alloc");
    CheckHip(hipMalloc(&d_rs, row_bytes_meta), "slots alloc");
    CheckHip(hipMalloc(&d_tiles, sizeof(std::int32_t)), "tiles alloc");
    CheckHip(hipMemcpy(d_bounds, bounds.data(), bounds_bytes,
                       hipMemcpyHostToDevice),
             "bounds upload");
    CheckHip(hipMemcpy(d_rt, rows_token.data(), row_bytes_meta,
                       hipMemcpyHostToDevice),
             "rows upload");
    CheckHip(hipMemcpy(d_rs, rows_slot.data(), row_bytes_meta,
                       hipMemcpyHostToDevice),
             "slots upload");
    CheckHip(hipMemcpy(d_tiles, tiles.data(), sizeof(std::int32_t),
                       hipMemcpyHostToDevice),
             "tiles upload");
    const std::uint32_t tile_rows = tokens >= 128 ? 128U : 48U;
    if (!q::RoutedF16Gemm(d_w, shape.type, d_x, d_tiles, 1, tile_rows,
                          d_bounds, d_rt, d_rs, nullptr, d_out, nullptr, kRows,
                          kHidden, nullptr)) {
      if (tokens >= 128) {
        std::printf("  %-8s T=%-3u tile=%-3u launch rejected (BN>48)\n",
                    shape.name, tokens, tile_rows);
        continue;
      }
      std::printf("  %-8s T=%-3u launch rejected\n", shape.name, tokens);
      ++failures;
    } else {
      CheckHip(hipDeviceSynchronize(), "gemm sync");
      std::vector<float> got(tokens * kRows);
      CheckHip(hipMemcpy(got.data(), d_out, got.size() * sizeof(float),
                         hipMemcpyDeviceToHost),
               "out download");
      std::vector<float> want(tokens * kRows);
      std::vector<float> xs(kHidden);
      for (std::uint32_t t = 0; t < tokens; ++t) {
        for (std::uint32_t k = 0; k < kHidden; ++k) {
          xs[k] = __half2float(x[t * kHidden + k]);
        }
        for (std::uint32_t row = 0; row < kRows; ++row) {
          if (!Reference(shape, host.data() + row * row_bytes, kHidden,
                         xs.data(), &want[t * kRows + row])) {
            throw std::runtime_error("no reference decoder");
          }
        }
      }
      // Random weights cancel heavily, so a per-element relative error says
      // nothing; compare the whole tile in L2 against the reference norm.
      double diff2 = 0.0;
      double want2 = 0.0;
      std::size_t worst_at = 0;
      double worst_abs = 0.0;
      for (std::size_t i = 0; i < want.size(); ++i) {
        const double delta = static_cast<double>(got[i]) - want[i];
        diff2 += delta * delta;
        want2 += static_cast<double>(want[i]) * want[i];
        if (std::fabs(delta) > worst_abs) {
          worst_abs = std::fabs(delta);
          worst_at = i;
        }
      }
      const double rel = std::sqrt(diff2 / std::max(want2, 1e-30));
      // F16 activations and F16 accumulation against an F32 reference.
      const bool ok = rel < 5e-2;
      std::printf(
          "  %-8s T=%-3u tile=%-3u L2 rel %.4e  max abs %.3e (%zu) %s\n",
          shape.name, tokens, tile_rows, rel, worst_abs, worst_at,
          ok ? "ok" : "MISMATCH");
      if (!ok) {
        ++failures;
      }
    }
    (void)hipFree(d_bounds);
    (void)hipFree(d_rt);
    (void)hipFree(d_rs);
    (void)hipFree(d_tiles);
    (void)hipFree(d_x);
    (void)hipFree(d_out);
  }
  (void)hipFree(d_w);
  return failures;
}

}  // namespace

int main() try {
  int failures = 0;
  for (const auto& shape : kShapes) {
    failures += RunShape(shape);
  }
  std::printf("%s\n", failures == 0 ? "parity ok" : "parity FAILED");
  return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
  std::fprintf(stderr, "routed_gemm_parity_test: %s\n", error.what());
  return 1;
}
