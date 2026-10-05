#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/hip/heap_placement.hpp"
#include "src/core/hip/weight_upload.hpp"
#include "src/core/platform/pages.hpp"
#include "src/core/platform/prefault.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#endif
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

#include "qfn_mmq.h"

namespace gufo::hip {
namespace {

void ReleaseWeightRegions(std::vector<QwenGpuWeightRegion>& regions) noexcept {
  for (auto& region : regions) {
    if (region.owns_device_allocation && region.device_data != nullptr) {
      (void)hipFree(region.device_data);
    } else if (region.host_copy != nullptr) {
      (void)hipHostUnregister(region.host_copy);
      platform::ReleaseCommittedPages(region.host_copy, region.size);
    }
    region = {};
  }
}

[[nodiscard]] hipError_t MapRegisteredRegion(
    const core::GgufMappedRegion& source, QwenGpuWeightRegion& destination,
    bool& source_was_read) {
  source_was_read = false;
  void* host_copy =
      platform::CloneMappedWeightsToAnonymousPages(source.data, source.size);
  if (host_copy == nullptr)
    return hipErrorOutOfMemory;
  source_was_read = true;
  const auto unmap = [&](void* pointer) {
    platform::ReleaseCommittedPages(pointer, source.size);
  };
  std::unique_ptr<void, decltype(unmap)> owned_copy(host_copy, unmap);
  const auto register_error = hipHostRegister(
      host_copy, source.size, hipHostRegisterMapped | hipHostRegisterReadOnly);
  if (register_error != hipSuccess)
    return register_error;

  void* device_data = nullptr;
  const auto pointer_error =
      hipHostGetDevicePointer(&device_data, host_copy, 0);
  if (pointer_error != hipSuccess) {
    (void)hipHostUnregister(host_copy);
    return pointer_error;
  }

  destination = {.host_data = source.data,
                 .device_data = device_data,
                 .host_copy = owned_copy.release(),
                 .size = source.size};
  return hipSuccess;
}

/// Device-resident placement: one hipMalloc copy in (on this APU target)
/// carve-out memory, filled from the cached mapping or by FillDeviceRegions.
[[nodiscard]] hipError_t AllocateDeviceRegion(
    const core::GgufMappedRegion& source, QwenGpuWeightRegion& destination) {
  void* device_data = nullptr;
  const auto status = hipMalloc(&device_data, source.size);
  if (status != hipSuccess)
    return status;
  destination = {.host_data = source.data,
                 .device_data = device_data,
                 .host_copy = nullptr,
                 .size = source.size,
                 .owns_device_allocation = true};
  return hipSuccess;
}

/// Streams one region from the read-only mapping into device memory. Where
/// the platform can drop the mapping's pages as it goes, eight workers each
/// copy 64 MiB chunks through their own pinned staging buffer, so a shard in
/// the page cache uploads at memory speed. A hipMemcpy straight from the
/// pageable mapping pins it page by page and runs at a small fraction of
/// that. Elsewhere one staging buffer streams the prefaulted mapping; that is
/// the fallback for sources WeightUpload cannot read, such as in-memory images
/// without a file.
[[nodiscard]] hipError_t CopyMappedRegion(const core::GgufMappedRegion& source,
                                          void* device_data) {
  const auto free_staging = [](void* pointer) { (void)hipHostFree(pointer); };
  using Staging = std::unique_ptr<void, decltype(free_staging)>;
  if (platform::CanVisitMappedChunks()) {
    constexpr std::size_t kChunkBytes = 64ULL << 20;
    constexpr std::size_t kWorkers = 8;
    std::vector<Staging> staging;
    staging.reserve(kWorkers);
    for (std::size_t i = 0; i < kWorkers; ++i)
      staging.emplace_back(nullptr, free_staging);
    std::atomic<hipError_t> failure{hipSuccess};
    platform::VisitMappedChunks(
        source.data, source.size, kChunkBytes, kWorkers,
        [&](std::size_t worker, std::size_t offset, const void* chunk,
            std::size_t size) {
          auto& buffer = staging[worker];
          if (!buffer) {
            void* pointer = nullptr;
            if (const auto status = hipHostMalloc(&pointer, kChunkBytes);
                status != hipSuccess) {
              failure.store(status, std::memory_order_relaxed);
              return false;
            }
            buffer.reset(pointer);
          }
          std::memcpy(buffer.get(), chunk, size);
          const auto status =
              hipMemcpy(static_cast<std::uint8_t*>(device_data) + offset,
                        buffer.get(), size, hipMemcpyHostToDevice);
          if (status != hipSuccess) {
            failure.store(status, std::memory_order_relaxed);
            return false;
          }
          return true;
        });
    return failure.load();
  }

  platform::PrefaultMappedRange(source.data, source.size);

  constexpr std::size_t kChunkBytes = 16ULL << 20;
  void* staging = nullptr;
  hipError_t status = hipHostMalloc(&staging, kChunkBytes);
  if (status != hipSuccess)
    return status;
  Staging owned_staging(staging, free_staging);

  const auto chunks =
      source.size / kChunkBytes + (source.size % kChunkBytes != 0);
  for (std::size_t index = 0; index < chunks; ++index) {
    const auto offset = index * kChunkBytes;
    const auto bytes = std::min(kChunkBytes, source.size - offset);
    std::memcpy(staging, static_cast<const std::uint8_t*>(source.data) + offset,
                bytes);
    status = hipMemcpy(static_cast<std::uint8_t*>(device_data) + offset,
                       staging, bytes, hipMemcpyHostToDevice);
    if (status != hipSuccess)
      return status;
  }
  return hipSuccess;
}

/// Fills device-resident regions from their shard files without touching the
/// mappings. Copying through a mapping faults the whole shard into host memory
/// first; on Windows those pages also stay in the process working set, so a
/// 16 GiB target needs as much free RAM beside the GPU carve-out. WeightUpload
/// keeps host residency at its staging buffers.
[[nodiscard]] bool FillDeviceRegions(
    std::span<const core::GgufMappedRegion> sources,
    std::span<QwenGpuWeightRegion> destinations,
    std::span<const std::size_t> indices, std::string* error_msg) {
  std::string upload_error;
  if (auto uploader = WeightUpload::Create(sources, &upload_error)) {
    for (const auto index : indices) {
      if (!uploader->Copy(static_cast<std::uint32_t>(index), 0,
                          sources[index].size, destinations[index].device_data,
                          error_msg))
        break;
    }
    // Finish waits for every queued copy, also after a failed Copy, so the
    // destinations stay valid until the uploader is gone.
    return uploader->Finish(error_msg);
  }
  for (const auto index : indices) {
    const auto status =
        CopyMappedRegion(sources[index], destinations[index].device_data);
    if (status != hipSuccess) {
      if (error_msg != nullptr) {
        *error_msg = "Failed to upload GGUF shard " + std::to_string(index) +
                     ": " + hipGetErrorString(status);
      }
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool CreateWeightRegions(
    const core::GgufReader& reader,
    std::vector<QwenGpuWeightRegion>& weight_regions, std::string* error_msg) {
  const auto source_regions = reader.GetMappedRegions();
  if (source_regions.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF reader has no mapped weight regions";
    }
    return false;
  }

  weight_regions.resize(source_regions.size());
  std::vector<std::size_t> device_regions;
  for (std::size_t i = 0; i < source_regions.size(); ++i) {
    const auto& source = source_regions[i];
    auto& destination = weight_regions[i];
    hipError_t map_error;
    try {
      // Device memory is not on the kernel's page LRU, so compaction never
      // migrates it. A registered anonymous copy is: every migration makes
      // the driver evict and restore the GPU mapping, and kernels reading it
      // stall for seconds.
      // Read through the page cache where the platform allows it, so a
      // warm reload does not wait on the disk; otherwise (Windows) the
      // bounded file pipeline fills the region without touching the mapping.
      std::size_t free_bytes = 0;
      std::size_t total_bytes = 0;
      if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess &&
          HeapFitsAboveReserve(free_bytes, source.size) &&
          AllocateDeviceRegion(source, destination) == hipSuccess) {
        if (platform::CanVisitMappedChunks())
          map_error = CopyMappedRegion(source, destination.device_data);
        else {
          map_error = hipSuccess;
          device_regions.push_back(i);
        }
      } else {
        (void)hipGetLastError();
        destination = {};
        bool source_was_read = false;
        map_error = MapRegisteredRegion(source, destination, source_was_read);
        if (map_error != hipSuccess) {
          // The device-resident upload preserves single-copy residency
          // wherever registration is refused.
          destination = {};
          map_error = AllocateDeviceRegion(source, destination);
          if (map_error == hipSuccess) {
            if (source_was_read) {
              // A successful clone has already read the mapping. Reuse those
              // cached bytes if registration fails instead of rereading the
              // shard with direct I/O. When cloning is unavailable (Windows),
              // keep the bounded file pipeline without touching the mapping.
              map_error = CopyMappedRegion(source, destination.device_data);
            } else {
              device_regions.push_back(i);
            }
          }
        }
      }
    } catch (const std::exception& e) {
      ReleaseWeightRegions(weight_regions);
      if (error_msg)
        *error_msg = e.what();
      return false;
    }
    if (map_error != hipSuccess || destination.device_data == nullptr) {
      ReleaseWeightRegions(weight_regions);
      if (error_msg != nullptr) {
        *error_msg = "Failed to make GGUF shard " + std::to_string(i) +
                     " GPU-visible: " + hipGetErrorString(map_error);
      }
      return false;
    }
  }
  if (!device_regions.empty() &&
      !FillDeviceRegions(source_regions, weight_regions, device_regions,
                         error_msg)) {
    ReleaseWeightRegions(weight_regions);
    return false;
  }
  return true;
}

[[nodiscard]] bool RemapTensor(
    models::QwenTensorRef& tensor,
    std::span<const QwenGpuWeightRegion> regions) noexcept {
  if (tensor.empty()) {
    return true;
  }
  const auto tensor_address = reinterpret_cast<std::uintptr_t>(tensor.data);
  for (const auto& region : regions) {
    const auto region_address =
        reinterpret_cast<std::uintptr_t>(region.host_data);
    if (tensor_address >= region_address) {
      const auto offset = tensor_address - region_address;
      const std::size_t encoded_bytes = tensor.EncodedSizeBytes();
      if (offset < region.size && encoded_bytes != 0 &&
          encoded_bytes <= region.size - offset) {
        tensor.data =
            static_cast<const std::uint8_t*>(region.device_data) + offset;
        tensor.available_bytes = region.size - offset;
        return true;
      }
    }
  }
  return false;
}

}  // namespace

QwenGpuModel::QwenGpuModel(
    std::shared_ptr<const core::GgufReader> reader,
    models::QwenModelWeights weights,
    std::shared_ptr<const tokenization::QwenTokenizer> tokenizer,
    std::vector<QwenGpuWeightRegion> weight_regions,
    std::shared_ptr<models::qwen::vision::Encoder> vision)
    : reader_(std::move(reader)),
      weights_(std::move(weights)),
      tokenizer_(std::move(tokenizer)),
      weight_regions_(std::move(weight_regions)),
      vision_(std::move(vision)) {}

QwenGpuModel::~QwenGpuModel() {
  for (void* allocation : prefill_copy_allocations_) {
    if (allocation != nullptr) {
      (void)hipFree(allocation);
    }
  }
  ReleaseWeightRegions(weight_regions_);
}

std::size_t QwenGpuModel::GetResidentBytes() const noexcept {
  std::size_t total = vision_ ? vision_->ResidentBytes() : 0;
  for (const auto& region : weight_regions_) {
    if (region.size > std::numeric_limits<std::size_t>::max() - total) {
      return std::numeric_limits<std::size_t>::max();
    }
    total += region.size;
  }
  if (prefill_copy_bytes_ >
      std::numeric_limits<std::size_t>::max() - total) {
    return std::numeric_limits<std::size_t>::max();
  }
  return total + prefill_copy_bytes_;
}

const models::QwenTensorRef* QwenGpuModel::FindPrefillCopy(
    const void* data) const noexcept {
  const auto found = prefill_copies_.find(data);
  return found == prefill_copies_.end() ? nullptr : &found->second;
}

bool QwenGpuModel::BuildPrefillCopies(std::string* error_msg) {
  // Opt-out without rebuilding (benchmarking / debugging).
  if (const char* disable = std::getenv("GUFO_QWEN_PREFILL_BF16_COPIES");
      disable != nullptr && std::string_view(disable) == "0") {
    return true;
  }
  // Q6_K dense projections only. MoE routed experts keep packed quantized
  // storage for the routed GEMM, and the output head plus embeddings run on
  // decode-only paths where the originals stay optimal.
  std::vector<models::QwenTensorRef*> candidates;
  for (auto& layer : weights_.layers) {
    models::QwenTensorRef* dense[] = {
        &layer.attn_qkv,     &layer.attn_gate,      &layer.ssm_out,
        &layer.attn_q,       &layer.attn_k,         &layer.attn_v,
        &layer.attn_output,  &layer.ffn_gate_shexp, &layer.ffn_up_shexp,
        &layer.ffn_down_shexp,
    };
    for (auto* tensor : dense) {
      if (tensor->type == core::GgmlType::kQ6_K && !tensor->empty() &&
          tensor->num_elements % 256 == 0) {
        candidates.push_back(tensor);
      }
    }
  }
  if (candidates.empty()) {
    return true;
  }
  std::atomic<std::size_t> next{0};
  std::atomic<bool> failure{false};
  std::string first_error;
  std::mutex mutex;
  // Host-side Q6_K -> Q8_0 requant, one 256-element superblock at a
  // time so only a small stack staging buffer is needed. Matches the GGUF
  // block_q8_0 layout ({half d; int8 qs[32]}, amax/127) the prefill W8A8
  // kernels consume.
  struct Q80Block {
    __half d;
    std::int8_t qs[32];
  };
  static_assert(sizeof(Q80Block) == 34, "block_q8_0 must be 34 bytes");
  auto convert_one = [&](models::QwenTensorRef* tensor) {
    const std::size_t elements = tensor->num_elements;
    const std::size_t dst_bytes = (elements / 32) * sizeof(Q80Block);
    std::vector<Q80Block> host_q8(elements / 32);
    {
      float staging[256];
      const auto* src_blocks =
          static_cast<const std::uint8_t*>(tensor->data);
      constexpr std::size_t kQ6KBytes = 210;
      for (std::size_t s = 0, ns = elements / 256; s < ns; ++s) {
        ::gufo::quant::DequantizeQ6_K(src_blocks + s * kQ6KBytes, staging,
                                      256);
        for (std::size_t b = 0; b < 8; ++b) {
          float amax = 0.0F;
          for (std::size_t i = 0; i < 32; ++i) {
            amax = std::max(amax, std::fabs(staging[b * 32 + i]));
          }
          const float d = amax / 127.0F;
          const float id = d != 0.0F ? 1.0F / d : 0.0F;
          Q80Block* out = &host_q8[s * 8 + b];
          out->d = __float2half_rn(d);
          for (std::size_t i = 0; i < 32; ++i) {
            out->qs[i] =
                static_cast<std::int8_t>(std::lrintf(staging[b * 32 + i] * id));
          }
        }
      }
    }
    void* device = nullptr;
    if (hipMalloc(&device, dst_bytes) != hipSuccess) {
      std::lock_guard<std::mutex> lock(mutex);
      if (!failure.exchange(true)) {
        first_error = "Q8_0 prefill copy allocation failed";
      }
      return;
    }
    if (hipMemcpy(device, host_q8.data(), dst_bytes, hipMemcpyHostToDevice) !=
        hipSuccess) {
      (void)hipFree(device);
      std::lock_guard<std::mutex> lock(mutex);
      if (!failure.exchange(true)) {
        first_error = "Q8_0 prefill copy upload failed";
      }
      return;
    }
    std::lock_guard<std::mutex> lock(mutex);
    if (failure.load()) {
      (void)hipFree(device);
      return;
    }
    prefill_copy_allocations_.push_back(device);
    prefill_copies_[tensor->data] = models::QwenTensorRef{
        .data = device,
        .type = core::GgmlType::kQ8_0,
        .num_elements = elements,
    };
    prefill_copy_bytes_ += dst_bytes;
  };
  {
    std::vector<std::jthread> workers;
    const std::size_t worker_count =
        std::min<std::size_t>(16, candidates.size());
    for (std::size_t w = 0; w + 1 < worker_count; ++w) {
      workers.emplace_back([&] {
        while (!failure.load(std::memory_order_relaxed)) {
          const auto index = next.fetch_add(1, std::memory_order_relaxed);
          if (index >= candidates.size()) {
            break;
          }
          convert_one(candidates[index]);
        }
      });
    }
    while (!failure.load(std::memory_order_relaxed)) {
      const auto index = next.fetch_add(1, std::memory_order_relaxed);
      if (index >= candidates.size()) {
        break;
      }
      convert_one(candidates[index]);
    }
  }
  if (failure.load()) {
    for (void* allocation : prefill_copy_allocations_) {
      (void)hipFree(allocation);
    }
    prefill_copy_allocations_.clear();
    prefill_copies_.clear();
    prefill_copy_bytes_ = 0;
    if (error_msg != nullptr) {
      *error_msg = std::move(first_error);
    }
    return false;
  }
  return true;
}

std::shared_ptr<const QwenGpuModel> QwenGpuModel::CreateFromGguf(
    std::shared_ptr<const core::GgufReader> reader, std::string* error_msg,
    std::shared_ptr<models::qwen::vision::Encoder> vision) {
  if (reader == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF reader must not be null";
    }
    return nullptr;
  }

  if (!tokenization::QwenChatTemplate::ValidateGgufTemplate(*reader,
                                                            error_msg)) {
    return nullptr;
  }

  auto weights_opt = models::QwenModelWeights::LoadFromGguf(*reader, error_msg);
  if (!weights_opt.has_value()) {
    return nullptr;
  }

  if (weights_opt->config.IsMoE()) {
    // The routed-expert GEMMs run on the vendored llama.cpp-derived MMQ
    // kernels; bind their device context before first use.
    if (qfn_mmq_init(0) != 0) {
      if (error_msg != nullptr) {
        *error_msg = "Failed to initialize the MoE quant kernel backend";
      }
      return nullptr;
    }
  }

  auto tokenizer =
      tokenization::QwenTokenizer::CreateFromGguf(*reader, error_msg);
  if (!tokenizer)
    return nullptr;

  std::vector<QwenGpuWeightRegion> weight_regions;
  if (!CreateWeightRegions(*reader, weight_regions, error_msg)) {
    return nullptr;
  }

  const auto remap = [&](models::QwenTensorRef& tensor) {
    return RemapTensor(tensor, weight_regions);
  };
  bool remapped = remap(weights_opt->token_embd) &&
                  remap(weights_opt->output_norm) && remap(weights_opt->output);
  for (auto& layer : weights_opt->layers) {
    remapped = remapped && remap(layer.attn_norm) && remap(layer.attn_q) &&
               remap(layer.attn_k) && remap(layer.attn_v) &&
               remap(layer.attn_output) && remap(layer.attn_q_norm) &&
               remap(layer.attn_k_norm) && remap(layer.attn_qkv) &&
               remap(layer.attn_gate) && remap(layer.ssm_out) &&
               remap(layer.ssm_conv1d) && remap(layer.ssm_alpha) &&
               remap(layer.ssm_beta) && remap(layer.ssm_a) &&
               remap(layer.ssm_dt) && remap(layer.ssm_norm) &&
                remap(layer.ffn_norm) && remap(layer.ffn_gate) &&
                remap(layer.ffn_up) && remap(layer.ffn_down) &&
                remap(layer.ffn_gate_inp) && remap(layer.ffn_gate_inp_shexp) &&
                remap(layer.ffn_gate_exps) && remap(layer.ffn_up_exps) &&
                remap(layer.ffn_down_exps) && remap(layer.ffn_gate_shexp) &&
                remap(layer.ffn_up_shexp) && remap(layer.ffn_down_shexp);
  }
  if (!remapped) {
    ReleaseWeightRegions(weight_regions);
    if (error_msg != nullptr) {
      *error_msg = "A Qwen tensor does not belong to any mapped GGUF shard";
    }
    return nullptr;
  }

  std::shared_ptr<const tokenization::QwenTokenizer> shared_tokenizer(
      std::move(tokenizer));
  auto model = std::make_shared<QwenGpuModel>(
      std::move(reader), std::move(*weights_opt), std::move(shared_tokenizer),
      std::move(weight_regions), std::move(vision));
  if (!model->BuildPrefillCopies(error_msg)) {
    return nullptr;
  }
  return model;
}

std::unique_ptr<QwenGpuExecutor> QwenGpuExecutor::Create(
    std::shared_ptr<const QwenGpuModel> model, std::string* error_msg,
    std::uint32_t max_context, QwenExecutionPolicy policy) {
  if (model == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "Qwen GPU model must not be null";
    }
    return nullptr;
  }

  const auto& config = model->GetConfig();
  const std::uint32_t model_context =
      config.context_length > 0 ? config.context_length : max_context;
  if (max_context == 0 || max_context > model_context) {
    if (error_msg != nullptr) {
      *error_msg = "Requested GPU context exceeds the model context length";
    }
    return nullptr;
  }

  return std::make_unique<QwenGpuExecutor>(std::move(model), max_context,
                                           policy);
}

std::unique_ptr<QwenGpuExecutor> QwenGpuExecutor::CreateFromGguf(
    std::shared_ptr<const core::GgufReader> reader, std::string* error_msg,
    std::uint32_t max_context, QwenExecutionPolicy policy) {
  auto model = QwenGpuModel::CreateFromGguf(std::move(reader), error_msg);
  if (model == nullptr) {
    return nullptr;
  }
  return Create(std::move(model), error_msg, max_context, policy);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
