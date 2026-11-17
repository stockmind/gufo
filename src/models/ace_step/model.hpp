#ifndef GUFO_MODELS_ACE_STEP_MODEL_HPP_
#define GUFO_MODELS_ACE_STEP_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::ace_step {

constexpr int kTextWidth = 1024;       // Qwen3-0.6B hidden size
constexpr int kConditionWidth = 2048;  // condition encoder width
constexpr int kLatentChannels = 64;    // VAE latent channels (25 frames / s)
constexpr int kTimbreFrames = 750;     // reference frames without audio

/// Qwen3-0.6B text encoder of ACE-Step 1.5 (ComfyUI `qwen_0.6b_ace15`,
/// Qwen3-Embedding-0.6B): causal, 28 layers, output after the final norm.
/// BF16 weights and GEMMs with an FP32 residual stream.
class TextEncoder {
public:
  [[nodiscard]] static std::unique_ptr<TextEncoder> Load(
      const std::filesystem::path& safetensors, std::string* error);
  ~TextEncoder();
  TextEncoder(const TextEncoder&) = delete;
  TextEncoder& operator=(const TextEncoder&) = delete;

  /// Hidden states [ids, 1024] after the final norm.
  bool Encode(std::span<const std::uint32_t> ids, std::vector<float>* states,
              std::string* error);
  /// Token embeddings [ids, 1024] (the lyric input of the condition encoder).
  bool Embed(std::span<const std::uint32_t> ids, std::vector<float>* embeds,
             std::string* error);
  [[nodiscard]] std::size_t weight_bytes() const;

private:
  struct Impl;
  explicit TextEncoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

/// Intermediates of one decoder call for oracle comparison (FP32, row-major).
struct DitProbes {
  std::vector<float> patch;   // [tokens, 2560] after proj_in
  std::vector<float> layer0;  // residual after layer 0
  std::vector<float> layer31;
};

/// The ACE-Step 1.5 XL turbo diffusion model (ComfyUI
/// `acestep_v1.5_xl_turbo_bf16`): condition encoder (text projector, lyric
/// and timbre encoders) and the 32-layer DiT decoder.
class Dit {
public:
  [[nodiscard]] static std::unique_ptr<Dit> Load(
      const std::filesystem::path& safetensors, std::string* error);
  ~Dit();
  Dit(const Dit&) = delete;
  Dit& operator=(const Dit&) = delete;

  /// Packed cross-attention states [lyric rows + 1 + text rows, 2048] from
  /// the text states [text_rows, 1024], lyric token embeddings [lyric_rows,
  /// 1024] and timbre reference latents [timbre_rows, 64].
  bool EncodeConditions(std::span<const float> text, int text_rows,
                        std::span<const float> lyric, int lyric_rows,
                        std::span<const float> timbre, int timbre_rows,
                        std::vector<float>* conditions, std::string* error);

  /// Fixes the conditions of the following Velocity calls: the packed states
  /// [rows, 2048] (projected into every layer's cross-attention keys and
  /// values once) and the context latents [frames, 128] (source latents and
  /// chunk mask).
  bool SetConditions(std::span<const float> conditions, int rows,
                     std::span<const float> context, int frames,
                     std::string* error);

  /// Flow velocity [frames, 64] at time t for x_t [frames, 64]. With
  /// `unconditional` the cross-attention reads the model's null condition
  /// (`null_condition_emb`) instead of the set conditions, for
  /// classifier-free guidance.
  bool Velocity(std::span<const float> xt, float t, std::vector<float>* v,
                DitProbes* probes, std::string* error,
                bool unconditional = false);

  /// 5 Hz audio codes (0..63999) of latents [frames, 64]: the audio
  /// tokenizer (acoustic projection, attention pooler over windows of 5
  /// frames, FSQ quantizer as ComfyUI / gufo's CodeHints define the
  /// codebook). The last window is padded with the first frames of
  /// `silence` [>= 4, 64] as the official `tokenize` does; ceil(frames / 5)
  /// codes.
  bool Tokenize(std::span<const float> latents, int frames,
                std::span<const float> silence, std::vector<int>* codes,
                std::string* error);

  /// Source latents [frames, 64] from 5 Hz audio codes (0..63999): FSQ
  /// codebook, quantizer projection and the detokenizer (5 latents per
  /// code). Missing codes are padded with code 35847 as ComfyUI does.
  bool CodeHints(std::span<const int> codes, int frames,
                 std::vector<float>* hints, std::string* error);

  [[nodiscard]] std::size_t weight_bytes() const;
  [[nodiscard]] double last_velocity_ms() const;

private:
  struct Impl;
  explicit Dit(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

struct CodeSampling {
  float cfg{2.0F};
  float temperature{0.85F};
  float top_p{0.9F};
  std::uint64_t seed{0};
};

/// The 5 Hz audio-code language model of ACE-Step 1.5: a Qwen3 decoder
/// (0.6B, 1.7B or 4B; width, heads, layers and FFN size are read from the
/// weights) with 64000 audio-code tokens from id 151669, tied embeddings
/// unless the file has `lm_head.weight`. Samples codes with
/// classifier-free guidance between a conditional and an unconditional
/// prompt (two KV caches).
class CodeLm {
public:
  static constexpr std::uint32_t kFirstCode = 151669;
  static constexpr int kCodes = 64000;

  [[nodiscard]] static std::unique_ptr<CodeLm> Load(
      const std::filesystem::path& safetensors, std::string* error);
  ~CodeLm();
  CodeLm(const CodeLm&) = delete;
  CodeLm& operator=(const CodeLm&) = delete;

  /// `count` codes (0..63999) continuing both prompts. `progress` (may be
  /// empty) gets the number of codes so far; returning false cancels.
  bool Generate(std::span<const std::uint32_t> conditional,
                std::span<const std::uint32_t> unconditional, int count,
                const CodeSampling& sampling, std::vector<int>* codes,
                const std::function<bool(int)>& progress, std::string* error);
  [[nodiscard]] std::size_t weight_bytes() const;
  /// Raw audio-code logits of the first step of the last Generate
  /// ([sequences, 64000]: conditional, then unconditional), for checks.
  [[nodiscard]] const std::vector<float>& first_logits() const;
  [[nodiscard]] double last_codes_per_second() const;

private:
  struct Impl;
  explicit CodeLm(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

/// The first `steps` values of MusicTimesteps (sampler.hpp): the turbo
/// schedule (shift 3, 8 steps) without the final 0.
std::vector<float> TurboTimesteps(int steps, float shift);

}  // namespace gufo::models::ace_step

#endif  // GUFO_MODELS_ACE_STEP_MODEL_HPP_
