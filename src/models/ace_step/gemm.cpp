#include "src/models/ace_step/gemm.hpp"

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <array>
#include <map>
#include <tuple>
#include <utility>

namespace gufo::models::ace_step {
namespace {

constexpr std::size_t kWorkspaceBytes = 32U << 20U;
constexpr int kCandidates = 16;

bool Fail(std::string* error, const char* operation, int status) {
  if (error != nullptr)
    *error = std::string("ACE-Step hipBLASLt ") + operation +
             " failed: " + std::to_string(status);
  return false;
}

struct Plan {
  hipblasLtMatmulDesc_t op{};
  hipblasLtMatrixLayout_t weight{}, input{}, output{};
  hipblasLtMatmulAlgo_t algo{};
  std::size_t workspace{0};
  ~Plan() {
    if (output != nullptr)
      (void)hipblasLtMatrixLayoutDestroy(output);
    if (input != nullptr)
      (void)hipblasLtMatrixLayoutDestroy(input);
    if (weight != nullptr)
      (void)hipblasLtMatrixLayoutDestroy(weight);
    if (op != nullptr)
      (void)hipblasLtMatmulDescDestroy(op);
  }
};

}  // namespace

struct LinearGemm::Impl {
  ~Impl() {
    plans.clear();
    if (workspace != nullptr)
      (void)hipFree(workspace);
    for (hipEvent_t event : events)
      if (event != nullptr)
        (void)hipEventDestroy(event);
    if (handle != nullptr)
      (void)hipblasLtDestroy(handle);
  }

  Plan* Resolve(const std::uint16_t* weight, const std::uint16_t* bias,
                const std::uint16_t* input, std::uint16_t* out, int tokens,
                int rows, int columns, std::string* error);
  bool Launch(Plan& plan, const hipblasLtMatmulAlgo_t& algo,
              const std::uint16_t* weight, const std::uint16_t* bias,
              const std::uint16_t* input, std::uint16_t* out);

  hipblasLtHandle_t handle{};
  hipStream_t stream{};
  void* workspace{nullptr};
  hipEvent_t events[2]{};
  std::map<std::tuple<int, int, int, bool>, std::unique_ptr<Plan>> plans;
  hipblasStatus_t last_status{HIPBLAS_STATUS_SUCCESS};
};

bool LinearGemm::Impl::Launch(Plan& plan, const hipblasLtMatmulAlgo_t& algo,
                              const std::uint16_t* weight,
                              const std::uint16_t* bias,
                              const std::uint16_t* input, std::uint16_t* out) {
  if (bias != nullptr) {
    last_status = hipblasLtMatmulDescSetAttribute(
        plan.op, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    if (last_status != HIPBLAS_STATUS_SUCCESS)
      return false;
  }
  const float alpha = 1, beta = 0;
  last_status =
      hipblasLtMatmul(handle, plan.op, &alpha, weight, plan.weight, input,
                      plan.input, &beta, out, plan.output, out, plan.output,
                      &algo, workspace, kWorkspaceBytes, stream);
  return last_status == HIPBLAS_STATUS_SUCCESS;
}

Plan* LinearGemm::Impl::Resolve(const std::uint16_t* weight,
                                const std::uint16_t* bias,
                                const std::uint16_t* input, std::uint16_t* out,
                                int tokens, int rows, int columns,
                                std::string* error) {
  const auto key = std::make_tuple(tokens, rows, columns, bias != nullptr);
  if (const auto found = plans.find(key); found != plans.end())
    return found->second.get();
  auto plan = std::make_unique<Plan>();
  hipblasStatus_t status =
      hipblasLtMatmulDescCreate(&plan->op, HIPBLAS_COMPUTE_32F, HIP_R_32F);
  const hipblasOperation_t transpose = HIPBLAS_OP_T;
  if (status == HIPBLAS_STATUS_SUCCESS)
    status = hipblasLtMatmulDescSetAttribute(
        plan->op, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose));
  if (status == HIPBLAS_STATUS_SUCCESS && bias != nullptr) {
    const hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
    const hipDataType bias_type = HIP_R_16BF;
    status = hipblasLtMatmulDescSetAttribute(
        plan->op, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
    if (status == HIPBLAS_STATUS_SUCCESS)
      status = hipblasLtMatmulDescSetAttribute(
          plan->op, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type,
          sizeof(bias_type));
    if (status == HIPBLAS_STATUS_SUCCESS)
      status = hipblasLtMatmulDescSetAttribute(
          plan->op, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
  }
  if (status == HIPBLAS_STATUS_SUCCESS)
    status = hipblasLtMatrixLayoutCreate(&plan->weight, HIP_R_16BF, columns,
                                         rows, columns);
  if (status == HIPBLAS_STATUS_SUCCESS)
    status = hipblasLtMatrixLayoutCreate(&plan->input, HIP_R_16BF, columns,
                                         tokens, columns);
  if (status == HIPBLAS_STATUS_SUCCESS)
    status = hipblasLtMatrixLayoutCreate(&plan->output, HIP_R_16BF, rows,
                                         tokens, rows);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    Fail(error, "plan", status);
    return nullptr;
  }
  hipblasLtMatmulPreference_t preference{};
  std::array<hipblasLtMatmulHeuristicResult_t, kCandidates> candidates{};
  int count = 0;
  status = hipblasLtMatmulPreferenceCreate(&preference);
  if (status == HIPBLAS_STATUS_SUCCESS) {
    const std::size_t limit = kWorkspaceBytes;
    status = hipblasLtMatmulPreferenceSetAttribute(
        preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit,
        sizeof(limit));
    if (status == HIPBLAS_STATUS_SUCCESS)
      status = hipblasLtMatmulAlgoGetHeuristic(
          handle, plan->op, plan->weight, plan->input, plan->output,
          plan->output, preference, kCandidates, candidates.data(), &count);
    (void)hipblasLtMatmulPreferenceDestroy(preference);
  }
  if (status != HIPBLAS_STATUS_SUCCESS || count <= 0) {
    Fail(error, "heuristic", status);
    return nullptr;
  }
  // Time every candidate on the caller's buffers; keep the fastest. A pass
  // in which no candidate launches is retried once after draining the stream
  // (seen intermittently on gfx1151 / Windows).
  float best = -1;
  int skipped = 0;
  for (int c = 0; c < count || (best < 0 && c < 2 * count); ++c) {
    if (c == count)
      (void)hipStreamSynchronize(stream);
    const auto& candidate = candidates[c % count];
    if (candidate.state != HIPBLAS_STATUS_SUCCESS ||
        candidate.workspaceSize > kWorkspaceBytes) {
      ++skipped;
      continue;
    }
    if (!Launch(*plan, candidate.algo, weight, bias, input, out))
      continue;
    (void)hipEventRecord(events[0], stream);
    bool ok = true;
    for (int run = 0; run < 3 && ok; ++run)
      ok = Launch(*plan, candidate.algo, weight, bias, input, out);
    (void)hipEventRecord(events[1], stream);
    float ms = 0;
    if (!ok || hipEventSynchronize(events[1]) != hipSuccess ||
        hipEventElapsedTime(&ms, events[0], events[1]) != hipSuccess)
      continue;
    if (best < 0 || ms < best) {
      best = ms;
      plan->algo = candidate.algo;
      plan->workspace = candidate.workspaceSize;
    }
  }
  if (best < 0) {
    const std::string operation =
        "tuning (no usable algorithm of " + std::to_string(count) + ", " +
        std::to_string(skipped) + " skipped) for " + std::to_string(tokens) +
        "x" + std::to_string(rows) + "x" + std::to_string(columns) +
        ", last device error " + hipGetErrorString(hipGetLastError()) +
        ", last status";
    Fail(error, operation.c_str(), static_cast<int>(last_status));
    return nullptr;
  }
  Plan* result = plan.get();
  plans.emplace(key, std::move(plan));
  return result;
}

LinearGemm::LinearGemm() : impl_(std::make_unique<Impl>()) {}
LinearGemm::~LinearGemm() = default;

bool LinearGemm::Init(hipStream_t stream, std::string* error) {
  impl_->stream = stream;
  if (const auto status = hipblasLtCreate(&impl_->handle);
      status != HIPBLAS_STATUS_SUCCESS)
    return Fail(error, "handle", status);
  if (hipMalloc(&impl_->workspace, kWorkspaceBytes) != hipSuccess)
    return Fail(error, "workspace", 0);
  for (hipEvent_t& event : impl_->events)
    if (hipEventCreate(&event) != hipSuccess)
      return Fail(error, "event", 0);
  return true;
}

bool LinearGemm::Run(const std::uint16_t* weight, const std::uint16_t* bias,
                     const std::uint16_t* input, std::uint16_t* out, int tokens,
                     int rows, int columns, std::string* error) {
  for (int first = 0; first < tokens; first += kRowChunk) {
    const int count = std::min(kRowChunk, tokens - first);
    const std::uint16_t* chunk_input =
        input + static_cast<std::size_t>(first) * columns;
    std::uint16_t* chunk_out = out + static_cast<std::size_t>(first) * rows;
    Plan* plan = impl_->Resolve(weight, bias, chunk_input, chunk_out, count,
                                rows, columns, error);
    if (plan == nullptr)
      return false;
    if (!impl_->Launch(*plan, plan->algo, weight, bias, chunk_input, chunk_out))
      return Fail(error, "matmul", 0);
  }
  return true;
}

}  // namespace gufo::models::ace_step
