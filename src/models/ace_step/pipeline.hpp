#ifndef GUFO_MODELS_ACE_STEP_PIPELINE_HPP_
#define GUFO_MODELS_ACE_STEP_PIPELINE_HPP_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/ace_step/audio_io.hpp"
#include "src/models/ace_step/preview.hpp"
#include "src/models/ace_step/sampler.hpp"

namespace gufo::models::ace_step {

/// Files of one ACE-Step 1.5 XL installation (turbo, sft or base).
struct ModelPaths {
  std::filesystem::path dit;        // BF16 DiT, ComfyUI / gufo layout
  std::filesystem::path text;       // qwen_0.6b_ace15.safetensors
  std::filesystem::path vae;        // Diffusers Oobleck vae safetensors
  std::filesystem::path tokenizer;  // Qwen3 tokenizer.json
  std::filesystem::path silence;    // silence_latent.safetensors [frames, 64]
  std::filesystem::path lm;         // optional 5 Hz LM (0.6B / 1.7B / 4B)
  DitVariant variant{DitVariant::kTurbo};

  /// Resolves the files below `root`: `dit.safetensors`,
  /// `text_encoder.safetensors`, `vae.safetensors`, `tokenizer.json` and
  /// `silence_latent.safetensors` (links into the ComfyUI / Hugging Face
  /// downloads are fine), and the optional 5 Hz LM `lm.safetensors`. The
  /// variant comes from the DiT file (see Resolve). False and `error` when
  /// a required file is missing.
  [[nodiscard]] static bool FromRoot(const std::filesystem::path& root,
                                     ModelPaths* paths, std::string* error);

  /// FromRoot with overrides. `dit`: "" (root/dit.safetensors), "turbo" /
  /// "sft" / "base" (root/dit-<variant>.safetensors; "turbo" falls back to
  /// dit.safetensors) or a file path. `lm`: "" (root/lm.safetensors when
  /// present), "none" (no LM), "0.6b" / "1.7b" / "4b"
  /// (root/lm-<size>.safetensors) or a file path. `variant`: "" (from the
  /// DiT file's `gufo.variant` / `model_version` metadata, else its file
  /// name, else turbo) or "turbo" / "sft" / "base".
  [[nodiscard]] static bool Resolve(const std::filesystem::path& root,
                                    std::string_view dit, std::string_view lm,
                                    std::string_view variant, ModelPaths* paths,
                                    std::string* error);
};

/// One music request (official ACE-Step 1.5 prompt format and tasks).
struct MusicRequest {
  std::string caption;         // style tags / description
  std::string lyrics;          // "[Instrumental]" when empty
  std::string language{"en"};  // lyric language code
  std::string bpm;             // "" -> N/A
  std::string keyscale;        // e.g. "A minor", "" -> N/A
  std::string timesignature;   // "4", "3", ... "" -> N/A
  double duration{30.0};       // seconds, 1..600 (text2music; source tasks
                               // take the source length)
  std::uint64_t seed{42};
  int steps{0};                // 0: the variant's default (8 / 50 / 32)
  float shift{0.0F};           // 0: the variant's default (3.0)
  float guidance_scale{0.0F};  // 0: the variant's default (1 / 7 / 7);
                               // turbo always runs without guidance
  float cfg_interval_start{0.0F};
  float cfg_interval_end{1.0F};
  /// Plan the song with the 5 Hz LM first (when the LM is loaded), as
  /// ComfyUI's `generate_audio_codes` default does; text2music only.
  bool audio_codes{true};
  float lm_cfg{2.0F};
  float lm_temperature{0.85F};
  float lm_top_p{0.9F};

  // Tasks (sampler.hpp): everything but text2music needs `source_audio`.
  MusicTask task{MusicTask::kText2Music};
  /// Planar stereo 48 kHz [2][samples]; empty when absent.
  std::vector<float> source_audio;
  /// Planar stereo 48 kHz timbre reference; empty = none (silence).
  std::vector<float> reference_audio;
  double repaint_start{0.0};  // seconds; negative extends in front
  double repaint_end{-1.0};   // seconds; < 0 = end of source, beyond it
                              // extends (outpainting)
  std::string repaint_mode{"balanced"};  // conservative | balanced | aggressive
  float repaint_strength{0.5F};          // balanced only, 0..1
  float cover_strength{1.0F};        // audio_cover_strength, 0..1
  float cover_noise_strength{0.0F};  // 0..1, start from the renoised source
  std::string track;                     // extract / lego stem name
  std::vector<std::string> track_classes;  // complete
  /// Draw the source latent as `latent_dist.sample()` (the official
  /// pipeline) instead of using the mean.
  bool vae_sample{true};
};

struct MusicTelemetry {
  double text_seconds{0};
  double encode_seconds{0};  // VAE encoding of source / reference audio
  double condition_seconds{0};
  double diffusion_seconds{0};
  double vae_seconds{0};
  double lm_seconds{0};
  int codes{0};
  int frames{0};
  int steps{0};
  int text_tokens{0};
  int lyric_tokens{0};
  bool guided{false};
};

/// Progress callback: fraction 0..1 and a short phase name. Returning false
/// cancels the generation.
using MusicProgress = std::function<bool(double, std::string_view)>;

/// The music pipeline: tokenizer, Qwen3-0.6B text encoder, condition
/// encoder, DiT sampler (Euler; APG classifier-free guidance for the sft /
/// base checkpoints), the Oobleck VAE decoder and, for source and
/// reference audio, its encoder stay loaded (about 12 GB of device
/// memory); the optional 5 Hz LM is loaded for each request that plans
/// audio codes and released again.
class MusicPipeline {
public:
  [[nodiscard]] static std::unique_ptr<MusicPipeline> Load(
      const ModelPaths& paths, std::string* error);
  ~MusicPipeline();
  MusicPipeline(const MusicPipeline&) = delete;
  MusicPipeline& operator=(const MusicPipeline&) = delete;

  [[nodiscard]] DitVariant variant() const;
  [[nodiscard]] bool has_lm() const;

  /// Planar stereo audio [2][samples] at 48 kHz, peak-normalized to <= 1.
  /// With a `preview` sink, every `preview->every`-th diffusion step (and
  /// the last one) decodes a short window of the clean estimate x0 (see
  /// MusicCleanEstimate) while `preview->wanted()` is true and the previous
  /// preview is at least `min_interval_seconds` old.
  bool Generate(const MusicRequest& request, std::vector<float>* audio,
                MusicTelemetry* telemetry, const MusicProgress& progress,
                std::string* error,
                const MusicPreviewSink* preview = nullptr);

private:
  struct Impl;
  explicit MusicPipeline(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

/// Text prompts of the official pipeline (exposed for tests): the DiT text
/// prompt with the task instruction and the duration in seconds.
std::string MusicTextPrompt(const MusicRequest& request,
                            std::string_view instruction, double duration);
std::string MusicTextPrompt(const MusicRequest& request);
std::string MusicLyricPrompt(const MusicRequest& request);
/// 5 Hz LM prompts: open assistant turn after the metadata "think" block;
/// the unconditional prompt is "NO USER INPUT" with an empty think block.
std::string MusicLmPrompt(const MusicRequest& request);
std::string MusicLmUnconditionalPrompt();

/// Standard normal noise [count] from `seed` (gufo's own generator).
std::vector<float> MusicNoise(std::uint64_t seed, std::size_t count);

}  // namespace gufo::models::ace_step

#endif  // GUFO_MODELS_ACE_STEP_PIPELINE_HPP_
