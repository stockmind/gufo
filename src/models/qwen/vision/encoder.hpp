#ifndef GUFO_MODELS_QWEN_VISION_ENCODER_HPP_
#define GUFO_MODELS_QWEN_VISION_ENCODER_HPP_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/image.hpp"

namespace gufo::models::qwen::vision {

/// How a Qwen language trunk reaches its vision sidecar. The projector output
/// width is the trunk's embedding length; text-only architectures return
/// std::nullopt.
struct VisionSidecarPlan {
  std::uint32_t output_width{0};
  /// qwen35 (Qwen3.8-27B) keeps a canonical mmproj-BF16.gguf beside the target.
  /// qwen35moe models share a directory with other quantizations, so discovery
  /// by filename is ambiguous and an explicit --mmproj is required.
  bool discover_beside_target{false};
};

[[nodiscard]] constexpr std::optional<VisionSidecarPlan> PlanVisionSidecar(
    std::string_view architecture, std::uint32_t embedding_length) noexcept {
  if (embedding_length == 0)
    return std::nullopt;
  if (architecture == "qwen35")
    return VisionSidecarPlan{embedding_length, true};
  if (architecture == "qwen35moe")
    return VisionSidecarPlan{embedding_length, false};
  return std::nullopt;
}

/// Shared Qwen ViT operator graph. Each model supplies and validates its own
/// projector/output width; language-model state never lives here.
class Encoder {
public:
  class Embedding {
  public:
    ~Embedding();
    Embedding(const Embedding&) = delete;
    Embedding& operator=(const Embedding&) = delete;
    [[nodiscard]] const float* data() const noexcept { return data_; }
    [[nodiscard]] std::uint32_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::uint32_t width() const noexcept { return width_; }

  private:
    friend class Encoder;
    Embedding() = default;
    float* data_{nullptr};
    std::uint32_t rows_{0};
    std::uint32_t width_{0};
  };

  using Observer =
      std::function<void(std::string_view, std::span<const float>)>;
  using CancellationCheck = std::function<bool()>;
  Encoder(const std::filesystem::path& path, std::uint32_t output_width);
  /// An explicit sidecar is required to exist. Otherwise discover the canonical
  /// sidecar beside the target or its quantization subdirectory.
  [[nodiscard]] static std::shared_ptr<Encoder> Open(
      const std::filesystem::path& target, const std::filesystem::path& sidecar,
      std::uint32_t output_width);
  ~Encoder();
  Encoder(const Encoder&) = delete;
  Encoder& operator=(const Encoder&) = delete;
  [[nodiscard]] const std::string& identity() const;
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  /// Serialized by the owning model scheduler. Pixel grid is already resized.
  /// Weights upload on the first image; scratch is bounded by image dimensions.
  [[nodiscard]] std::shared_ptr<const Embedding> Encode(
      const core::Image& image, const Observer& observer = {},
      const CancellationCheck& is_cancelled = {});

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen::vision
#endif
