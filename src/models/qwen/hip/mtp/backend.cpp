#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/qwen/hip/mtp.hpp"

namespace gufo::hip {
namespace {

/// Candidate width of an MTP sampled proposal. The verification residual only
/// subtracts draft mass inside this row, so keep it wide enough to cover the
/// head's temperature-scaled support while staying cheap to transfer.
constexpr std::size_t kMtpCandidateTopK = 64;

constexpr std::array<std::uint8_t, 8> kMtpDraftPersistentMagic = {
    'G', 'M', 'T', 'D', 'P', 'U', '0', '1'};
constexpr std::uint32_t kMtpDraftPersistentVersion = 1;
constexpr std::size_t kMtpDraftPersistentHeaderBytes = 64;
constexpr std::uint32_t kMtpDraftPrimedFlag = 1U << 0U;

template<typename T>
  requires(std::is_unsigned_v<T>)
void PutLittleEndian(std::span<std::uint8_t> destination, std::size_t offset,
                     T value) {
  if (offset > destination.size() || sizeof(T) > destination.size() - offset) {
    throw std::length_error("MTP draft persistent header is truncated");
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
    throw std::invalid_argument("MTP draft persistent header is truncated");
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
    throw std::overflow_error("MTP draft persistent size overflows");
  }
  return left + right;
}

[[nodiscard]] std::size_t PersistentSizeFromU64(std::uint64_t value) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("MTP draft persistent size overflows");
  }
  return static_cast<std::size_t>(value);
}

/// Draft-visible continuation of the MTP graph: the GPU KV copy plus the
/// host anchor features that pair the next draft input with its target row.
class QwenMtpDraftSnapshot final : public speculative::IDraftBackendSnapshot {
public:
  QwenMtpDraftSnapshot(std::unique_ptr<QwenMtpGpuSnapshot> gpu_snapshot,
                       std::vector<float> target_hidden,
                       std::vector<float> last_anchor_hidden, bool primed)
      : gpu_snapshot(std::move(gpu_snapshot)),
        target_hidden(std::move(target_hidden)),
        last_anchor_hidden(std::move(last_anchor_hidden)),
        primed(primed) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return (gpu_snapshot != nullptr ? gpu_snapshot->PayloadBytes() : 0) +
           target_hidden.size() * sizeof(float) +
           last_anchor_hidden.size() * sizeof(float) + sizeof(bool);
  }

  [[nodiscard]] std::size_t PersistentPayloadBytes() const override {
    if (gpu_snapshot == nullptr ||
        target_hidden.size() >
            std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t) ||
        last_anchor_hidden.size() >
            std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) {
      throw std::invalid_argument(
          "MTP draft snapshot persistent metadata is invalid");
    }
    return CheckedPersistentAdd(
        CheckedPersistentAdd(kMtpDraftPersistentHeaderBytes,
                             gpu_snapshot->PersistentPayloadBytes()),
        CheckedPersistentAdd(target_hidden.size() * sizeof(std::uint32_t),
                             last_anchor_hidden.size() * sizeof(std::uint32_t)));
  }

  [[nodiscard]] std::size_t SerializePersistent(
      std::span<std::uint8_t> destination) const override {
    const std::size_t expected_bytes = PersistentPayloadBytes();
    if (destination.size() != expected_bytes) {
      throw std::invalid_argument(
          "MTP draft persistent destination is invalid");
    }
    if (!primed && last_anchor_hidden.size() != 0)
      throw std::invalid_argument("MTP draft unprimed payload contains state");
    const std::size_t gpu_payload_bytes =
        gpu_snapshot->PersistentPayloadBytes();
    const std::size_t target_bytes = target_hidden.size() * sizeof(std::uint32_t);
    const std::size_t anchor_bytes =
        last_anchor_hidden.size() * sizeof(std::uint32_t);

    std::fill(destination.begin(), destination.end(), std::uint8_t{0});
    std::copy(kMtpDraftPersistentMagic.begin(), kMtpDraftPersistentMagic.end(),
              destination.begin());
    PutLittleEndian<std::uint32_t>(destination, 8, kMtpDraftPersistentVersion);
    PutLittleEndian<std::uint32_t>(
        destination, 12,
        static_cast<std::uint32_t>(kMtpDraftPersistentHeaderBytes));
    PutLittleEndian<std::uint32_t>(destination, 16,
                                   primed ? kMtpDraftPrimedFlag : 0U);
    PutLittleEndian<std::uint64_t>(
        destination, 24, static_cast<std::uint64_t>(target_hidden.size()));
    PutLittleEndian<std::uint64_t>(
        destination, 32, static_cast<std::uint64_t>(last_anchor_hidden.size()));
    PutLittleEndian<std::uint64_t>(
        destination, 40, static_cast<std::uint64_t>(gpu_payload_bytes));
    PutLittleEndian<std::uint64_t>(destination, 48,
                                   static_cast<std::uint64_t>(expected_bytes));
    PutLittleEndian<std::uint32_t>(destination, 56,
                                   gpu_snapshot->ValidContext());

    const std::size_t gpu_offset = kMtpDraftPersistentHeaderBytes;
    const std::size_t target_offset =
        CheckedPersistentAdd(gpu_offset, gpu_payload_bytes);
    const std::size_t anchor_offset =
        CheckedPersistentAdd(target_offset, target_bytes);
    const std::size_t written = gpu_snapshot->SerializePersistent(
        destination.subspan(gpu_offset, gpu_payload_bytes));
    if (written != gpu_payload_bytes) {
      throw std::runtime_error(
          "MTP GPU persistent serializer returned the wrong byte count");
    }
    for (std::size_t index = 0; index < target_hidden.size(); ++index) {
      PutLittleEndian<std::uint32_t>(
          destination, target_offset + index * sizeof(std::uint32_t),
          std::bit_cast<std::uint32_t>(target_hidden[index]));
    }
    for (std::size_t index = 0; index < last_anchor_hidden.size(); ++index) {
      PutLittleEndian<std::uint32_t>(
          destination, anchor_offset + index * sizeof(std::uint32_t),
          std::bit_cast<std::uint32_t>(last_anchor_hidden[index]));
    }
    if (CheckedPersistentAdd(anchor_offset, anchor_bytes) !=
        destination.size()) {
      throw std::logic_error(
          "MTP draft persistent serializer size mismatch");
    }
    return destination.size();
  }

  std::unique_ptr<QwenMtpGpuSnapshot> gpu_snapshot;
  std::vector<float> target_hidden;
  std::vector<float> last_anchor_hidden;
  bool primed{false};
};

/// Samples one token from the temperature-scaled top-k of `logits`, appending
/// the normalized candidate row the verifier needs for lossless rejection.
tokenization::TokenId SampleMtpTopK(
    std::span<const float> logits, float temperature, std::uint64_t* rng_state,
    std::vector<tokenization::TokenId>& candidate_ids,
    std::vector<float>& candidate_probabilities) {
  if (logits.empty() || !std::isfinite(temperature) || temperature <= 0.0F) {
    throw std::invalid_argument("MTP sampled proposal arguments are invalid");
  }
  const std::size_t vocab = logits.size();
  const std::size_t k = std::min(kMtpCandidateTopK, vocab);
  std::vector<std::uint32_t> order(vocab);
  std::iota(order.begin(), order.end(), 0U);
  std::partial_sort(order.begin(), order.begin() + k, order.end(),
                    [&](std::uint32_t lhs, std::uint32_t rhs) {
                      return logits[lhs] > logits[rhs];
                    });
  const double maximum = logits[order.front()];
  std::vector<double> probabilities(k);
  double sum = 0.0;
  for (std::size_t index = 0; index < k; ++index) {
    probabilities[index] = std::exp(
        (static_cast<double>(logits[order[index]]) - maximum) / temperature);
    sum += probabilities[index];
  }
  if (!(sum > 0.0) || !std::isfinite(sum)) {
    throw std::runtime_error("MTP sampled proposal has no probability mass");
  }
  const double uniform = static_cast<double>(sampling::Uniform(rng_state));
  double cumulative = 0.0;
  std::size_t chosen = k - 1;
  for (std::size_t index = 0; index < k; ++index) {
    cumulative += probabilities[index] / sum;
    if (uniform < cumulative) {
      chosen = index;
      break;
    }
  }
  for (std::size_t index = 0; index < k; ++index) {
    candidate_ids.push_back(order[index]);
    candidate_probabilities.push_back(
        static_cast<float>(probabilities[index] / sum));
  }
  return order[chosen];
}

}  // namespace

QwenMtpGpuDraftBackend::QwenMtpGpuDraftBackend(
    std::unique_ptr<QwenMtpGpuExecutor> executor, QwenMtpGpuDraftConfig config)
    : executor_(std::move(executor)),
      config_(config),
      target_hidden_(executor_->GetHiddenSize()) {}

std::unique_ptr<QwenMtpGpuDraftBackend> QwenMtpGpuDraftBackend::Create(
    std::shared_ptr<const QwenMtpGpuModel> model, QwenMtpGpuDraftConfig config,
    std::string* error_msg) {
  if (model == nullptr || config.max_draft_tokens == 0) {
    if (error_msg != nullptr) {
      *error_msg = "MTP GPU draft configuration is invalid";
    }
    return nullptr;
  }
  auto executor = QwenMtpGpuExecutor::Create(std::move(model),
                                             config.max_context, error_msg);
  if (executor == nullptr) {
    return nullptr;
  }
  executor->SetVisionInput(config.vision_input);
  return std::unique_ptr<QwenMtpGpuDraftBackend>(
      new QwenMtpGpuDraftBackend(std::move(executor), config));
}

std::unique_ptr<QwenMtpGpuDraftBackend> QwenMtpGpuDraftBackend::CreateFromGguf(
    std::string_view model_path,
    std::shared_ptr<const QwenGpuModel> target_model,
    QwenMtpGpuDraftConfig config, std::string* error_msg) {
  if (model_path.empty() || target_model == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "MTP GGUF path and target GPU model are required";
    }
    return nullptr;
  }
  auto reader_owner =
      core::GgufReader::OpenFile(std::string(model_path), error_msg);
  if (reader_owner == nullptr) {
    return nullptr;
  }
  std::shared_ptr<const core::GgufReader> reader(std::move(reader_owner));
  auto model = QwenMtpGpuModel::Create(std::move(reader),
                                       std::move(target_model), error_msg);
  if (model == nullptr) {
    return nullptr;
  }
  return Create(std::move(model), config, error_msg);
}

bool QwenMtpGpuDraftBackend::PrimeTargetContext(
    const speculative::DraftTargetContext& context) {
  Reset();
  if (context.prompt_tokens.empty() || context.hidden_size == 0 ||
      context.hidden_size != target_hidden_.size() ||
      context.prompt_hidden_states.size() !=
          context.prompt_tokens.size() * context.hidden_size) {
    last_error_ = "MTP prompt hidden-state shape is invalid";
    return false;
  }

  try {
    for (std::size_t index = 0; index + 1 < context.prompt_tokens.size();
         ++index) {
      const auto hidden = context.prompt_hidden_states.subspan(
          index * context.hidden_size, context.hidden_size);
      (void)executor_->ForwardTargetHidden(
          context.prompt_tokens[index + 1], hidden,
          static_cast<std::uint32_t>(index), false);
    }
    const std::size_t final_offset =
        (context.prompt_tokens.size() - 1) * context.hidden_size;
    const auto final_hidden =
        context.prompt_hidden_states.subspan(final_offset, context.hidden_size);
    std::ranges::copy(final_hidden, target_hidden_.begin());
    primed_ = true;
    return true;
  } catch (const std::exception& exception) {
    last_error_ = exception.what();
    Reset();
    return false;
  }
}

bool QwenMtpGpuDraftBackend::AppendTargetContext(
    const speculative::DraftTargetContext& context, std::uint32_t position) {
  if (!primed_ || proposal_active_ ||
      executor_->GetNextPosition() + 1 != position ||
      context.hidden_size != target_hidden_.size() ||
      context.prompt_hidden_states.size() !=
          context.prompt_tokens.size() * context.hidden_size) {
    return false;
  }
  for (std::size_t row = 0; row < context.prompt_tokens.size(); ++row) {
    (void)executor_->ForwardTargetHidden(
        context.prompt_tokens[row], target_hidden_,
        position + static_cast<std::uint32_t>(row) - 1, false);
    UpdateTargetHidden(context.prompt_hidden_states.subspan(
        row * context.hidden_size, context.hidden_size));
  }
  return true;
}

speculative::DraftProposal QwenMtpGpuDraftBackend::Propose(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens) {
  if (!primed_ || prompt_tokens.empty() || current_pos == 0) {
    throw std::logic_error("MTP GPU draft backend is not primed");
  }
  if (proposal_active_) {
    throw std::logic_error("MTP GPU proposal feedback is pending");
  }
  if (executor_->GetNextPosition() + 1 != current_pos) {
    throw std::logic_error("MTP GPU draft position is inconsistent");
  }

  speculative::DraftProposal proposal;
  proposal.start_pos = current_pos;
  const std::uint32_t count = std::min(max_tokens, config_.max_draft_tokens);
  if (count == 0) {
    return proposal;
  }

  proposal_checkpoint_ = executor_->GetNextPosition();
  proposal_input_ = prompt_tokens.back();
  proposal_target_hidden_ = target_hidden_;
  last_anchor_hidden_ = target_hidden_;
  committed_target_hidden_.clear();
  proposed_tokens_.clear();
  proposed_tokens_.reserve(count);

  auto token = executor_->ForwardTargetHidden(proposal_input_, target_hidden_,
                                              proposal_checkpoint_, true);
  proposed_tokens_.push_back(token);
  for (std::uint32_t index = 1; index < count; ++index) {
    token =
        executor_->ForwardFeedback(token, proposal_checkpoint_ + index, true);
    proposed_tokens_.push_back(token);
  }
  proposal.tokens = proposed_tokens_;
  proposal_active_ = true;
  return proposal;
}

speculative::DraftProposal QwenMtpGpuDraftBackend::ProposeSampled(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens, float temperature,
    std::uint64_t* rng_state) {
  if (!std::isfinite(temperature) || temperature <= 0.0F) {
    throw std::invalid_argument(
        "MTP sampled proposal temperature must be finite and positive");
  }
  if (rng_state == nullptr) {
    throw std::invalid_argument("MTP sampled proposal requires RNG state");
  }
  if (!primed_ || prompt_tokens.empty() || current_pos == 0) {
    throw std::logic_error("MTP GPU draft backend is not primed");
  }
  if (proposal_active_) {
    throw std::logic_error("MTP GPU proposal feedback is pending");
  }
  if (executor_->GetNextPosition() + 1 != current_pos) {
    throw std::logic_error("MTP GPU draft position is inconsistent");
  }

  speculative::DraftProposal proposal;
  proposal.start_pos = current_pos;
  const std::uint32_t count = std::min(max_tokens, config_.max_draft_tokens);
  if (count == 0) {
    return proposal;
  }

  proposal_checkpoint_ = executor_->GetNextPosition();
  proposal_input_ = prompt_tokens.back();
  proposal_target_hidden_ = target_hidden_;
  last_anchor_hidden_ = target_hidden_;
  committed_target_hidden_.clear();
  proposed_tokens_.clear();
  proposed_tokens_.reserve(count);
  proposal.candidates_per_token =
      std::min<std::size_t>(kMtpCandidateTopK, executor_->GetVocabSize());

  auto logits = executor_->ForwardTargetHiddenLogits(
      proposal_input_, target_hidden_, proposal_checkpoint_);
  auto token =
      SampleMtpTopK(logits, temperature, rng_state, proposal.candidate_ids,
                    proposal.candidate_probabilities);
  proposed_tokens_.push_back(token);
  for (std::uint32_t index = 1; index < count; ++index) {
    logits = executor_->ForwardFeedbackLogits(proposed_tokens_.back(),
                                              proposal_checkpoint_ + index);
    token =
        SampleMtpTopK(logits, temperature, rng_state, proposal.candidate_ids,
                      proposal.candidate_probabilities);
    proposed_tokens_.push_back(token);
  }
  proposal.tokens = proposed_tokens_;
  proposal_active_ = true;
  return proposal;
}

void QwenMtpGpuDraftBackend::AcceptFeedback(
    std::span<const tokenization::TokenId> accepted,
    tokenization::TokenId correction_token) {
  (void)correction_token;
  if (!proposal_active_ || accepted.size() > proposed_tokens_.size()) {
    throw std::logic_error("MTP GPU proposal feedback is invalid");
  }
  if (committed_target_hidden_.size() !=
      (accepted.size() + 1) * target_hidden_.size()) {
    throw std::logic_error("MTP feedback requires each committed target row");
  }

  // Pair input t+1 with target hidden t, as during prompt priming. Verification
  // has already updated target_hidden_ to the final committed row; it cannot
  // be reused for the earlier anchor. Accepted rows also need target features,
  // rather than the tentative draft hidden states used while proposing.
  executor_->Rewind(proposal_checkpoint_);
  (void)executor_->ForwardTargetHidden(proposal_input_, proposal_target_hidden_,
                                       proposal_checkpoint_, false);
  for (std::size_t index = 0; index < accepted.size(); ++index) {
    (void)executor_->ForwardTargetHidden(
        accepted[index],
        std::span<const float>(committed_target_hidden_)
            .subspan(index * target_hidden_.size(), target_hidden_.size()),
        proposal_checkpoint_ + static_cast<std::uint32_t>(index) + 1, false);
  }
  proposal_active_ = false;
  proposed_tokens_.clear();
  proposal_target_hidden_.clear();
  committed_target_hidden_.clear();
}

void QwenMtpGpuDraftBackend::UpdateTargetHidden(std::span<const float> hidden) {
  if (hidden.size() != target_hidden_.size()) {
    throw std::invalid_argument("MTP target hidden-state shape is invalid");
  }
  if (proposal_active_) {
    committed_target_hidden_.insert(committed_target_hidden_.end(),
                                    hidden.begin(), hidden.end());
  }
  std::ranges::copy(hidden, target_hidden_.begin());
}

void QwenMtpGpuDraftBackend::Reset() noexcept {
  executor_->Reset();
  std::ranges::fill(target_hidden_, 0.0F);
  proposed_tokens_.clear();
  proposal_target_hidden_.clear();
  committed_target_hidden_.clear();
  last_anchor_hidden_.clear();
  proposal_input_ = 0;
  proposal_checkpoint_ = 0;
  primed_ = false;
  proposal_active_ = false;
  last_error_.clear();
}

void QwenMtpGpuDraftBackend::DiscardPendingTargetContext(
    std::uint32_t position) {
  if (!primed_)
    return;
  // The draft executes eagerly while proposing, so rewind it to the target
  // frontier and restore the anchor hidden that pairs with `position`.
  proposal_active_ = false;
  proposed_tokens_.clear();
  proposal_target_hidden_.clear();
  committed_target_hidden_.clear();
  if (last_anchor_hidden_.size() == target_hidden_.size()) {
    std::ranges::copy(last_anchor_hidden_, target_hidden_.begin());
  }
  if (position == 0)
    return;
  const std::uint32_t checkpoint = position - 1;
  if (checkpoint <= executor_->GetNextPosition())
    executor_->Rewind(checkpoint);
}

std::size_t QwenMtpGpuDraftBackend::SnapshotPayloadBytes() const {
  const std::size_t gpu_bytes = executor_->SnapshotPayloadBytes();
  std::size_t bytes = CheckedPersistentAdd(
      gpu_bytes, target_hidden_.size() * sizeof(float));
  return CheckedPersistentAdd(
      bytes, last_anchor_hidden_.size() * sizeof(float) + sizeof(bool));
}

std::unique_ptr<speculative::IDraftBackendSnapshot>
QwenMtpGpuDraftBackend::Snapshot() const {
  if (proposal_active_ || committed_target_hidden_.size() != 0) {
    throw std::logic_error(
        "MTP snapshot requires a committed proposal boundary");
  }
  return std::make_unique<QwenMtpDraftSnapshot>(
      executor_->SaveSnapshot(), target_hidden_, last_anchor_hidden_, primed_);
}

void QwenMtpGpuDraftBackend::RestoreSnapshot(
    const speculative::IDraftBackendSnapshot& snapshot) {
  const auto* mtp_snapshot =
      dynamic_cast<const QwenMtpDraftSnapshot*>(&snapshot);
  if (mtp_snapshot == nullptr || mtp_snapshot->gpu_snapshot == nullptr)
    throw std::invalid_argument(
        "MTP draft snapshot is incompatible with the backend");
  const std::size_t hidden = executor_->GetHiddenSize();
  // The target anchor is always hidden-sized once primed; the cancellation
  // anchor is optional and only present after a proposal has been made, so an
  // empty value is a valid committed boundary.
  if (mtp_snapshot->target_hidden.size() != hidden ||
      (mtp_snapshot->last_anchor_hidden.size() != 0 &&
       mtp_snapshot->last_anchor_hidden.size() != hidden))
    throw std::invalid_argument("MTP draft snapshot has malformed anchors");
  executor_->RestoreSnapshot(*mtp_snapshot->gpu_snapshot);
  target_hidden_ = mtp_snapshot->target_hidden;
  last_anchor_hidden_ = mtp_snapshot->last_anchor_hidden;
  proposed_tokens_.clear();
  proposal_target_hidden_.clear();
  committed_target_hidden_.clear();
  proposal_input_ = 0;
  proposal_checkpoint_ = 0;
  primed_ = mtp_snapshot->primed;
  proposal_active_ = false;
}

void QwenMtpGpuDraftBackend::RestorePersistentSnapshot(
    std::span<const std::uint8_t> payload) {
  if (payload.size() < kMtpDraftPersistentHeaderBytes ||
      !std::equal(kMtpDraftPersistentMagic.begin(),
                  kMtpDraftPersistentMagic.end(), payload.begin()) ||
      GetLittleEndian<std::uint32_t>(payload, 8) !=
          kMtpDraftPersistentVersion ||
      GetLittleEndian<std::uint32_t>(payload, 12) !=
          kMtpDraftPersistentHeaderBytes) {
    throw std::invalid_argument("MTP draft persistent header is invalid");
  }
  const std::uint32_t flags = GetLittleEndian<std::uint32_t>(payload, 16);
  const std::size_t target_count =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 24));
  const std::size_t anchor_count =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 32));
  const std::size_t gpu_payload_bytes =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 40));
  const std::size_t total_bytes =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 48));
  const std::uint32_t valid_context =
      GetLittleEndian<std::uint32_t>(payload, 56);
  const std::size_t hidden = executor_->GetHiddenSize();

  if ((flags & ~kMtpDraftPrimedFlag) != 0 ||
      target_count != hidden ||
      (anchor_count != 0 && anchor_count != hidden) ||
      target_count > std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t) ||
      anchor_count > std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t) ||
      valid_context > config_.max_context ||
      GetLittleEndian<std::uint32_t>(payload, 20) != 0) {
    throw std::invalid_argument("MTP draft persistent metadata is incompatible");
  }
  const bool primed = (flags & kMtpDraftPrimedFlag) != 0;
  const std::size_t target_bytes = target_count * sizeof(std::uint32_t);
  const std::size_t anchor_bytes = anchor_count * sizeof(std::uint32_t);
  const std::size_t gpu_offset = kMtpDraftPersistentHeaderBytes;
  const std::size_t target_offset =
      CheckedPersistentAdd(gpu_offset, gpu_payload_bytes);
  const std::size_t anchor_offset =
      CheckedPersistentAdd(target_offset, target_bytes);
  if (CheckedPersistentAdd(anchor_offset, anchor_bytes) != payload.size() ||
      total_bytes != payload.size() ||
      target_offset > payload.size()) {
    throw std::invalid_argument(
        "MTP draft persistent payload size is invalid");
  }

  std::vector<float> target_hidden(target_count);
  std::vector<float> anchor_hidden(anchor_count);
  for (std::size_t index = 0; index < target_count; ++index) {
    target_hidden[index] = std::bit_cast<float>(
        GetLittleEndian<std::uint32_t>(payload,
                                       target_offset + index * sizeof(std::uint32_t)));
  }
  for (std::size_t index = 0; index < anchor_count; ++index) {
    anchor_hidden[index] = std::bit_cast<float>(
        GetLittleEndian<std::uint32_t>(payload,
                                       anchor_offset + index * sizeof(std::uint32_t)));
  }

  executor_->RestorePersistentSnapshot(
      payload.subspan(gpu_offset, gpu_payload_bytes));
  target_hidden_ = std::move(target_hidden);
  last_anchor_hidden_ = std::move(anchor_hidden);
  proposed_tokens_.clear();
  proposal_target_hidden_.clear();
  committed_target_hidden_.clear();
  proposal_input_ = 0;
  proposal_checkpoint_ = 0;
  primed_ = primed;
  proposal_active_ = false;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
