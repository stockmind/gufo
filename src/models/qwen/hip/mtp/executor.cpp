#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/hip/snapshot_transfer.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/mtp.hpp"
#include "src/models/qwen/hip/mtp/detail/allocation.hpp"
#include "src/models/qwen/hip/ops.hpp"

namespace gufo::hip {
namespace {

template<typename T>
void AllocateBuffer(T*& pointer, std::size_t elements) {
  pointer = static_cast<T*>(detail::AllocateDevice(elements * sizeof(T)));
}

constexpr std::array<std::uint8_t, 8> kMtpGpuPersistentMagic = {
    'G', 'M', 'T', 'P', 'P', 'U', '0', '1'};
constexpr std::uint32_t kMtpGpuPersistentVersion = 1;
constexpr std::size_t kMtpGpuPersistentHeaderBytes = 64;

template<typename T>
  requires(std::is_unsigned_v<T>)
void PutLittleEndian(std::span<std::uint8_t> destination, std::size_t offset,
                     T value) {
  if (offset > destination.size() || sizeof(T) > destination.size() - offset) {
    throw std::length_error("MTP GPU persistent header is truncated");
  }
  for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
    destination[offset + byte] =
        static_cast<std::uint8_t>(value >> (byte * 8U));
  }
}

template<typename T>
  requires(std::is_unsigned_v<T>)
[[nodiscard]] T GetLittleEndian(std::span<const std::uint8_t> source,
                                std::size_t offset) {
  if (offset > source.size() || sizeof(T) > source.size() - offset) {
    throw std::invalid_argument("MTP GPU persistent header is truncated");
  }
  T value = 0;
  for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
    value |= static_cast<T>(source[offset + byte]) << (byte * 8U);
  }
  return value;
}

[[nodiscard]] std::size_t CheckedPersistentAdd(std::size_t left,
                                               std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error("MTP GPU persistent size overflows");
  }
  return left + right;
}

[[nodiscard]] std::size_t PersistentSizeFromU64(std::uint64_t value) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("MTP GPU persistent size overflows");
  }
  return static_cast<std::size_t>(value);
}

}  // namespace

QwenMtpGpuSnapshot::~QwenMtpGpuSnapshot() {
  if (d_kv_f32_ != nullptr) {
    (void)hipFree(d_kv_f32_);
  }
  if (d_kv_f16_ != nullptr) {
    (void)hipFree(d_kv_f16_);
  }
}

std::size_t QwenMtpGpuSnapshot::PersistentPayloadBytes() const {
  return CheckedPersistentAdd(kMtpGpuPersistentHeaderBytes, payload_bytes_);
}

std::size_t QwenMtpGpuSnapshot::SerializePersistent(
    std::span<std::uint8_t> destination) const {
  const std::size_t expected_bytes = PersistentPayloadBytes();
  if (destination.size() != expected_bytes || kv_width_ == 0 ||
      valid_context_ > max_context_ ||
      (kv_width_ != 0 &&
       valid_context_ > std::numeric_limits<std::size_t>::max() / kv_width_)) {
    throw std::invalid_argument("MTP GPU snapshot metadata is malformed");
  }
  const std::size_t elements_per_plane =
      static_cast<std::size_t>(valid_context_) * kv_width_;
  const std::size_t f32_bytes = elements_per_plane * sizeof(float);
  const std::size_t f16_bytes = elements_per_plane * sizeof(std::uint16_t);
  if (f32_bytes > std::numeric_limits<std::size_t>::max() - f32_bytes ||
      payload_bytes_ != f32_bytes + f32_bytes ||
      (f32_bytes != 0 && (d_kv_f32_ == nullptr || d_kv_f16_ == nullptr))) {
    throw std::invalid_argument("MTP GPU snapshot payload is malformed");
  }

  std::fill(destination.begin(), destination.end(), std::uint8_t{0});
  std::copy(kMtpGpuPersistentMagic.begin(), kMtpGpuPersistentMagic.end(),
            destination.begin());
  PutLittleEndian<std::uint32_t>(destination, 8, kMtpGpuPersistentVersion);
  PutLittleEndian<std::uint32_t>(
      destination, 12,
      static_cast<std::uint32_t>(kMtpGpuPersistentHeaderBytes));
  PutLittleEndian<std::uint32_t>(destination, 16,
                                 static_cast<std::uint32_t>(kv_width_));
  PutLittleEndian<std::uint32_t>(destination, 20, max_context_);
  PutLittleEndian<std::uint32_t>(destination, 24, valid_context_);
  PutLittleEndian<std::uint64_t>(destination, 32,
                                 static_cast<std::uint64_t>(payload_bytes_));
  PutLittleEndian<std::uint64_t>(destination, 40,
                                 static_cast<std::uint64_t>(expected_bytes));

  if (elements_per_plane != 0) {
    SnapshotTransfer transfer;
    transfer.Copy(destination.data() + kMtpGpuPersistentHeaderBytes, d_kv_f32_,
                  f32_bytes);
    transfer.Copy(
        destination.data() + kMtpGpuPersistentHeaderBytes + f32_bytes,
        d_kv_f16_, f16_bytes);
  }
  return destination.size();
}

QwenMtpGpuExecutor::QwenMtpGpuExecutor(
    std::shared_ptr<const QwenMtpGpuModel> model, std::uint32_t max_context)
    : model_(std::move(model)),
      max_context_(max_context),
      h_last_hidden_(model_->GetConfig().hidden_size),
      h_logits_(model_->GetConfig().vocab_size) {
  try {
    Allocate();
    Reset();
  } catch (...) {
    Free();
    throw;
  }
}

QwenMtpGpuExecutor::~QwenMtpGpuExecutor() {
  if (stream_ != nullptr) {
    (void)hipStreamSynchronize(stream_);
  }
  Free();
}

std::unique_ptr<QwenMtpGpuExecutor> QwenMtpGpuExecutor::Create(
    std::shared_ptr<const QwenMtpGpuModel> model, std::uint32_t max_context,
    std::string* error_msg) {
  if (model == nullptr || max_context == 0 ||
      max_context > model->GetConfig().context_length) {
    if (error_msg != nullptr) {
      *error_msg = "MTP GPU executor context is invalid";
    }
    return nullptr;
  }
  try {
    return std::unique_ptr<QwenMtpGpuExecutor>(
        new QwenMtpGpuExecutor(std::move(model), max_context));
  } catch (const std::exception& exception) {
    if (error_msg != nullptr) {
      *error_msg = exception.what();
    }
    return nullptr;
  }
}

QwenGpuMemoryUsage QwenMtpGpuExecutor::EstimateMemoryUsage(
    const QwenMtpGpuModel& model, std::uint32_t max_context) noexcept {
  const auto& config = model.GetConfig();
  const std::size_t hidden = config.hidden_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t total_kv =
      static_cast<std::size_t>(config.num_key_value_heads) * max_context *
      config.head_dim;
  std::size_t bytes = 0;
  // target_hidden, embedding, hidden, normed, attn_out, ffn_out, feedback.
  bytes += hidden * sizeof(float) * 7;
  bytes += 2 * hidden * sizeof(float);     // fusion input
  bytes += 2 * attention * sizeof(float);  // packed q/gate
  bytes += attention * sizeof(float) * 3;  // q, gate, context
  bytes += kv * sizeof(float) * 2;         // k, v
  bytes += config.IsMoE() ? MoeScratchBytes(config, 1)
                          : config.intermediate_size * sizeof(float);
  bytes += config.vocab_size * sizeof(float);     // logits
  bytes += total_kv * sizeof(float) * 2;          // f32 kv (k and v)
  bytes += total_kv * sizeof(std::uint16_t) * 2;  // f16 kv (k and v)
  bytes += detail::DecodeAttentionScratchElements(config.num_attention_heads,
                                                  config.head_dim) *
           sizeof(float);
  bytes += sizeof(std::uint32_t);  // sampled token
  return {.request_state_bytes = bytes, .temporary_scratch_bytes = 0};
}

QwenGpuMemoryUsage QwenMtpGpuExecutor::GetMemoryUsage() const noexcept {
  return EstimateMemoryUsage(*model_, max_context_);
}

void QwenMtpGpuExecutor::Allocate() {
  const auto& config = model_->GetConfig();
  const std::size_t hidden = config.hidden_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t intermediate = config.intermediate_size;
  const std::size_t total_kv =
      static_cast<std::size_t>(config.num_key_value_heads) * max_context_ *
      config.head_dim;

  const auto stream_error = hipStreamCreate(&stream_);
  if (stream_error != hipSuccess) {
    throw std::runtime_error(std::string("MTP stream creation failed: ") +
                             hipGetErrorString(stream_error));
  }
  AllocateBuffer(d_target_hidden_, hidden);
  AllocateBuffer(d_embedding_, hidden);
  AllocateBuffer(d_fusion_, 2 * hidden);
  AllocateBuffer(d_hidden_, hidden);
  AllocateBuffer(d_normed_, hidden);
  AllocateBuffer(d_qg_, 2 * attention);
  AllocateBuffer(d_q_, attention);
  AllocateBuffer(d_k_, kv);
  AllocateBuffer(d_v_, kv);
  AllocateBuffer(d_gate_, attention);
  AllocateBuffer(d_context_, attention);
  AllocateBuffer(d_attn_out_, hidden);
  if (config.IsMoE()) {
    d_moe_ = static_cast<std::uint8_t*>(
        detail::AllocateDevice(MoeScratchBytes(config, 1)));
    moe_ = CarveMoeScratch(d_moe_, config, 1);
  } else {
    AllocateBuffer(d_ffn_act_, intermediate);
  }
  AllocateBuffer(d_ffn_out_, hidden);
  AllocateBuffer(d_feedback_hidden_, hidden);
  AllocateBuffer(d_logits_, config.vocab_size);
  AllocateBuffer(d_kv_cache_, 2 * total_kv);
  d_kv_cache_f16_ =
      detail::AllocateDevice(2 * total_kv * sizeof(std::uint16_t));
  AllocateBuffer(d_split_k_scratch_,
                 detail::DecodeAttentionScratchElements(
                     config.num_attention_heads, config.head_dim));
  AllocateBuffer(d_out_token_, 1);
}

void QwenMtpGpuExecutor::Free() noexcept {
  const auto free_buffer = [](auto*& pointer) {
    if (pointer != nullptr) {
      (void)hipFree(pointer);
      pointer = nullptr;
    }
  };
  free_buffer(d_target_hidden_);
  free_buffer(d_embedding_);
  free_buffer(d_fusion_);
  free_buffer(d_hidden_);
  free_buffer(d_normed_);
  free_buffer(d_qg_);
  free_buffer(d_q_);
  free_buffer(d_k_);
  free_buffer(d_v_);
  free_buffer(d_gate_);
  free_buffer(d_context_);
  free_buffer(d_attn_out_);
  free_buffer(d_ffn_act_);
  free_buffer(d_ffn_out_);
  free_buffer(d_moe_);
  moe_ = {};
  free_buffer(d_feedback_hidden_);
  free_buffer(d_logits_);
  free_buffer(d_kv_cache_);
  free_buffer(d_kv_cache_f16_);
  free_buffer(d_split_k_scratch_);
  free_buffer(d_out_token_);
  if (stream_ != nullptr) {
    (void)hipStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

void QwenMtpGpuExecutor::Reset() noexcept {
  // Attention only reads the valid prefix; new rows overwrite stale KV.
  next_position_ = 0;
}

void QwenMtpGpuExecutor::Rewind(std::uint32_t position) {
  if (position > next_position_) {
    throw std::out_of_range("MTP rewind position exceeds executed context");
  }
  next_position_ = position;
}

tokenization::TokenId QwenMtpGpuExecutor::ForwardTargetHidden(
    tokenization::TokenId input_token, std::span<const float> target_hidden,
    std::uint32_t position, bool compute_logits) {
  if (target_hidden.size() != model_->GetConfig().hidden_size) {
    throw std::invalid_argument("MTP target hidden width is invalid");
  }
  const auto copy_error = hipMemcpyAsync(d_target_hidden_, target_hidden.data(),
                                         target_hidden.size_bytes(),
                                         hipMemcpyHostToDevice, stream_);
  if (copy_error != hipSuccess) {
    throw std::runtime_error(std::string("MTP hidden copy failed: ") +
                             hipGetErrorString(copy_error));
  }
  return Run(input_token, d_target_hidden_, position, compute_logits);
}

tokenization::TokenId QwenMtpGpuExecutor::ForwardFeedback(
    tokenization::TokenId input_token, std::uint32_t position,
    bool compute_logits) {
  return Run(input_token, d_feedback_hidden_, position, compute_logits);
}

std::span<const float> QwenMtpGpuExecutor::ForwardTargetHiddenLogits(
    tokenization::TokenId input_token, std::span<const float> target_hidden,
    std::uint32_t position) {
  (void)ForwardTargetHidden(input_token, target_hidden, position, true);
  return CopyLastLogits();
}

std::span<const float> QwenMtpGpuExecutor::ForwardFeedbackLogits(
    tokenization::TokenId input_token, std::uint32_t position) {
  (void)ForwardFeedback(input_token, position, true);
  return CopyLastLogits();
}

tokenization::TokenId QwenMtpGpuExecutor::Run(tokenization::TokenId input_token,
                                              const float* hidden_input,
                                              std::uint32_t position,
                                              bool compute_logits) {
  if (position != next_position_ || position >= max_context_) {
    throw std::out_of_range("MTP GPU position is not sequential");
  }
  const auto& weights = model_->GetWeights();
  const auto& config = weights.config;
  const auto& layer = weights.layer;
  const std::size_t hidden = config.hidden_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t total_kv =
      static_cast<std::size_t>(config.num_key_value_heads) * max_context_ *
      config.head_dim;

  LaunchEmbeddingLookup(weights.token_embedding.data,
                        weights.token_embedding.type, input_token, d_embedding_,
                        hidden, stream_);
  if (vision_input_)
    vision_input_->Inject(d_embedding_, position + 1, 1, hidden, 1, stream_);
  LaunchRMSNorm(d_embedding_,
                static_cast<const float*>(weights.embedding_norm.data),
                d_fusion_, hidden, 1.0e-6F, stream_);
  LaunchRMSNorm(hidden_input,
                static_cast<const float*>(weights.hidden_norm.data),
                d_fusion_ + hidden, hidden, 1.0e-6F, stream_);
  LaunchGEMV(weights.fusion_projection.data, weights.fusion_projection.type,
             d_fusion_, d_hidden_, hidden, 2 * hidden, stream_,
             models::qwen::QwenGemmMode::kHipMtp);

  LaunchRMSNorm(d_hidden_, static_cast<const float*>(layer.attn_norm.data),
                d_normed_, hidden, 1.0e-6F, stream_);
  LaunchFusedQKVProjections(
      layer.attn_q.data, layer.attn_q.type, layer.attn_k.data,
      layer.attn_k.type, layer.attn_v.data, layer.attn_v.type, d_normed_, d_qg_,
      d_k_, d_v_, 2 * attention, kv, hidden, stream_);
  LaunchUnpackQG(d_qg_, d_q_, d_gate_, config.num_attention_heads,
                 config.head_dim, stream_);
  LaunchPerHeadRMSNorm(d_q_, static_cast<const float*>(layer.attn_q_norm.data),
                       d_q_, config.num_attention_heads, config.head_dim,
                       1.0e-6F, stream_);
  LaunchPerHeadRMSNorm(d_k_, static_cast<const float*>(layer.attn_k_norm.data),
                       d_k_, config.num_key_value_heads, config.head_dim,
                       1.0e-6F, stream_);
  LaunchRoPE(d_q_, d_k_, config.num_attention_heads, config.num_key_value_heads,
             config.head_dim, config.rotary_dim, position, config.rope_theta,
             stream_, vision_input_ ? vision_input_->rope() : nullptr);
  LaunchAttention(
      d_q_, d_k_, d_v_, d_gate_, d_kv_cache_, d_kv_cache_ + total_kv,
      d_kv_cache_f16_, static_cast<std::uint16_t*>(d_kv_cache_f16_) + total_kv,
      d_context_, 0, position, max_context_, config.num_attention_heads,
      config.num_key_value_heads, config.head_dim, stream_, d_split_k_scratch_);
  LaunchGEMV(layer.attn_output.data, layer.attn_output.type, d_context_,
             d_attn_out_, hidden, attention, stream_,
             models::qwen::QwenGemmMode::kHipMtp);
  LaunchResidualAdd(d_hidden_, d_attn_out_, d_hidden_, hidden, stream_);

  LaunchRMSNorm(d_hidden_, static_cast<const float*>(layer.ffn_norm.data),
                d_normed_, hidden, 1.0e-6F, stream_);
  if (config.IsMoE()) {
    // The MTP block of a MoE target is a MoE block too; it runs the same
    // single-token kernels as the target's decode step.
    ExecuteMoeDecodeStep(stream_, moe_, layer, config, d_normed_, d_ffn_out_);
  } else {
    LaunchFusedSwiGLUGEMV(layer.ffn_gate.data, layer.ffn_gate.type,
                          layer.ffn_up.data, layer.ffn_up.type, d_normed_,
                          d_ffn_act_, config.intermediate_size, hidden,
                          stream_);
    LaunchGEMV(layer.ffn_down.data, layer.ffn_down.type, d_ffn_act_, d_ffn_out_,
               hidden, config.intermediate_size, stream_,
               models::qwen::QwenGemmMode::kHipMtp);
  }
  LaunchResidualAdd(d_hidden_, d_ffn_out_, d_hidden_, hidden, stream_);
  LaunchRMSNorm(d_hidden_,
                static_cast<const float*>(weights.shared_head_norm.data),
                d_feedback_hidden_, hidden, 1.0e-6F, stream_);
  ++next_position_;

  if (!compute_logits) {
    return 0;
  }
  LaunchGEMV(weights.output.data, weights.output.type, d_feedback_hidden_,
             d_logits_, config.vocab_size, hidden, stream_,
             models::qwen::QwenGemmMode::kHipMtp);
  LaunchGPUArgmax(d_logits_, d_out_token_, config.vocab_size, stream_);
  tokenization::TokenId result = 0;
  const auto copy_error = hipMemcpyAsync(&result, d_out_token_, sizeof(result),
                                         hipMemcpyDeviceToHost, stream_);
  if (copy_error != hipSuccess || hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("MTP GPU result synchronization failed");
  }
  return CheckedSampleToken(result, config.vocab_size);
}

std::span<const float> QwenMtpGpuExecutor::CopyLastHidden() {
  const auto error = hipMemcpyAsync(h_last_hidden_.data(), d_feedback_hidden_,
                                    h_last_hidden_.size() * sizeof(float),
                                    hipMemcpyDeviceToHost, stream_);
  if (error != hipSuccess || hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("MTP hidden synchronization failed");
  }
  return h_last_hidden_;
}

std::span<const float> QwenMtpGpuExecutor::CopyLastLogits() {
  const auto error = hipMemcpyAsync(h_logits_.data(), d_logits_,
                                    h_logits_.size() * sizeof(float),
                                    hipMemcpyDeviceToHost, stream_);
  if (error != hipSuccess || hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("MTP logit synchronization failed");
  }
  return h_logits_;
}

std::size_t QwenMtpGpuExecutor::SnapshotPayloadBytes() const noexcept {
  const std::size_t kv_width =
      static_cast<std::size_t>(model_->GetConfig().num_key_value_heads) *
      model_->GetConfig().head_dim;
  const std::size_t elements_per_plane =
      static_cast<std::size_t>(next_position_) * kv_width;
  const std::size_t f32_bytes = elements_per_plane * sizeof(float);
  const std::size_t f16_bytes = elements_per_plane * sizeof(std::uint16_t);
  if (f32_bytes > std::numeric_limits<std::size_t>::max() - f32_bytes) {
    throw std::overflow_error("MTP snapshot size overflows");
  }
  return f32_bytes + f32_bytes + f16_bytes + f16_bytes;
}

std::unique_ptr<QwenMtpGpuSnapshot> QwenMtpGpuExecutor::SaveSnapshot() const {
  const std::size_t kv_width =
      static_cast<std::size_t>(model_->GetConfig().num_key_value_heads) *
      model_->GetConfig().head_dim;
  const std::size_t elements_per_plane =
      static_cast<std::size_t>(next_position_) * kv_width;
  const std::size_t f32_bytes = elements_per_plane * sizeof(float);
  const std::size_t f16_bytes = elements_per_plane * sizeof(std::uint16_t);

  auto snapshot =
      std::unique_ptr<QwenMtpGpuSnapshot>(new QwenMtpGpuSnapshot());
  snapshot->kv_width_ = kv_width;
  snapshot->max_context_ = max_context_;
  snapshot->valid_context_ = next_position_;
  snapshot->payload_bytes_ = 2 * f32_bytes + 2 * f16_bytes;

  if (elements_per_plane == 0) {
    return snapshot;
  }
  HIP_CHECK(hipMalloc(&snapshot->d_kv_f32_, 2 * f32_bytes));
  HIP_CHECK(hipMalloc(&snapshot->d_kv_f16_, 2 * f16_bytes));
  SnapshotTransfer transfer;
  transfer.Copy(snapshot->d_kv_f32_, d_kv_cache_, 2 * f32_bytes);
  transfer.Copy(snapshot->d_kv_f16_, d_kv_cache_f16_, 2 * f16_bytes);
  return snapshot;
}

void QwenMtpGpuExecutor::RestoreSnapshot(const QwenMtpGpuSnapshot& snapshot) {
  const std::size_t kv_width =
      static_cast<std::size_t>(model_->GetConfig().num_key_value_heads) *
      model_->GetConfig().head_dim;
  if (snapshot.kv_width_ != kv_width ||
      snapshot.max_context_ != max_context_ ||
      snapshot.valid_context_ > max_context_ ||
      (kv_width != 0 &&
       snapshot.valid_context_ > std::numeric_limits<std::size_t>::max() /
                                    kv_width)) {
    throw std::invalid_argument("MTP snapshot is incompatible with the executor");
  }
  const std::size_t elements_per_plane =
      static_cast<std::size_t>(snapshot.valid_context_) * kv_width;
  const std::size_t f32_bytes = elements_per_plane * sizeof(float);
  const std::size_t f16_bytes = elements_per_plane * sizeof(std::uint16_t);
  if (elements_per_plane != 0 &&
      (snapshot.d_kv_f32_ == nullptr || snapshot.d_kv_f16_ == nullptr)) {
    throw std::invalid_argument("MTP snapshot payload is incomplete");
  }
  if (elements_per_plane != 0) {
    SnapshotTransfer transfer;
    transfer.Copy(d_kv_cache_, snapshot.d_kv_f32_, 2 * f32_bytes,
                  hipMemcpyDeviceToDevice);
    transfer.Copy(d_kv_cache_f16_, snapshot.d_kv_f16_, 2 * f16_bytes,
                  hipMemcpyDeviceToDevice);
  }
  next_position_ = snapshot.valid_context_;
}

void QwenMtpGpuExecutor::RestorePersistentSnapshot(
    std::span<const std::uint8_t> payload) {
  if (payload.size() < kMtpGpuPersistentHeaderBytes ||
      !std::equal(kMtpGpuPersistentMagic.begin(), kMtpGpuPersistentMagic.end(),
                  payload.begin()) ||
      GetLittleEndian<std::uint32_t>(payload, 8) != kMtpGpuPersistentVersion ||
      GetLittleEndian<std::uint32_t>(payload, 12) !=
          kMtpGpuPersistentHeaderBytes) {
    throw std::invalid_argument("MTP GPU persistent header is invalid");
  }
  const std::size_t kv_width = GetLittleEndian<std::uint32_t>(payload, 16);
  const std::uint32_t max_context = GetLittleEndian<std::uint32_t>(payload, 20);
  const std::uint32_t valid_context = GetLittleEndian<std::uint32_t>(payload, 24);
  const std::size_t payload_bytes =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 32));
  const std::size_t total_bytes =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 40));

  const std::size_t expected_kv_width =
      static_cast<std::size_t>(model_->GetConfig().num_key_value_heads) *
      model_->GetConfig().head_dim;
  if (expected_kv_width > std::numeric_limits<std::uint32_t>::max() ||
      kv_width != expected_kv_width || max_context != max_context_ ||
      valid_context > max_context_ ||
      GetLittleEndian<std::uint32_t>(payload, 28) != 0 ||
      GetLittleEndian<std::uint32_t>(payload, 44) != 0 ||
      GetLittleEndian<std::uint32_t>(payload, 48) != 0 ||
      GetLittleEndian<std::uint32_t>(payload, 52) != 0 ||
      GetLittleEndian<std::uint32_t>(payload, 56) != 0 ||
      GetLittleEndian<std::uint32_t>(payload, 60) != 0) {
    throw std::invalid_argument("MTP GPU persistent metadata is incompatible");
  }
  const std::size_t elements_per_plane =
      static_cast<std::size_t>(valid_context) * kv_width;
  const std::size_t f32_bytes = elements_per_plane * sizeof(float);
  const std::size_t f16_bytes = elements_per_plane * sizeof(std::uint16_t);
  if (payload_bytes != 2 * f32_bytes + 2 * f16_bytes ||
      total_bytes != payload.size() ||
      CheckedPersistentAdd(kMtpGpuPersistentHeaderBytes, payload_bytes) !=
          payload.size()) {
    throw std::invalid_argument("MTP GPU persistent payload size is invalid");
  }
  if (elements_per_plane != 0) {
    SnapshotTransfer transfer;
    transfer.Copy(d_kv_cache_, payload.data() + kMtpGpuPersistentHeaderBytes,
                  2 * f32_bytes, hipMemcpyHostToDevice);
    transfer.Copy(d_kv_cache_f16_,
                  payload.data() + kMtpGpuPersistentHeaderBytes + 2 * f32_bytes,
                  2 * f16_bytes, hipMemcpyHostToDevice);
  }
  next_position_ = valid_context;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
