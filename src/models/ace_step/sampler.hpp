// Sampler mathematics of the ACE-Step 1.5 pipeline that needs no model and
// no GPU: task types and model variants, the timestep schedule, APG / CFG
// guidance, repaint masks and blends, the cover start and the waveform
// splice. Ported from the official ACE-Step 1.5 inference code
// (acestep/models/xl_base/modeling_acestep_v15_xl_base.py,
// acestep/core/generation/handler/*). Written with AI assistance (Claude
// Code).
#ifndef GUFO_MODELS_ACE_STEP_SAMPLER_HPP_
#define GUFO_MODELS_ACE_STEP_SAMPLER_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::ace_step {

/// Official task types (acestep/constants.py TASK_TYPES).
enum class MusicTask {
  kText2Music,
  kRepaint,
  kCover,
  kCoverNoFsq,
  kExtract,
  kLego,
  kComplete
};

/// The three XL checkpoints; same architecture, different samplers.
enum class DitVariant { kTurbo, kSft, kBase };

[[nodiscard]] std::string_view MusicTaskName(MusicTask task);
/// "text2music", "repaint", "cover", "cover-nofsq" (also "cover_nofsq"),
/// "extract", "lego", "complete"; case-insensitive.
[[nodiscard]] bool ParseMusicTask(std::string_view name, MusicTask* task);
[[nodiscard]] std::string_view DitVariantName(DitVariant variant);
[[nodiscard]] bool ParseDitVariant(std::string_view name, DitVariant* variant);
/// Finds "turbo" / "sft" / "base" in a file name or metadata value.
[[nodiscard]] bool GuessDitVariant(std::string_view text, DitVariant* variant);

/// Sampler defaults of the official UI (model_config.py): turbo 8 steps,
/// shift 3, no CFG; sft 50 steps, base 32 steps, both shift 3 and APG
/// guidance 7.
struct SamplerDefaults {
  int steps{8};
  float shift{3.0F};
  float guidance_scale{1.0F};
};
[[nodiscard]] SamplerDefaults DefaultSampler(DitVariant variant);

/// Task availability (TASK_TYPES_TURBO / TASK_TYPES_BASE): extract, lego and
/// complete exist on the pure base model only. Empty when allowed, else the
/// reason for a 400.
[[nodiscard]] std::string TaskUnsupportedReason(MusicTask task,
                                                DitVariant variant);
/// Every task but text2music needs source audio.
[[nodiscard]] bool TaskNeedsSource(MusicTask task);
/// Tasks whose source latents go through the FSQ tokenizer (`is_covers`).
[[nodiscard]] bool TaskIsCover(MusicTask task);
/// Tasks with a repaint range (chunk mask and repaint mask).
[[nodiscard]] bool TaskRepaints(MusicTask task);

/// The "# Instruction" line of the DiT text prompt (TASK_INSTRUCTIONS):
/// `track` is the stem for extract / lego, `track_classes` the stems for
/// complete (upper-cased, joined with " | ").
[[nodiscard]] std::string MusicInstruction(
    MusicTask task, std::string_view track,
    std::span<const std::string> track_classes);
[[nodiscard]] std::string DefaultMusicInstruction();

/// The official schedule: linspace(1, 0, steps + 1), then
/// t = shift * t / (1 + (shift - 1) * t) when shift != 1. `steps + 1`
/// values, the last one 0.
[[nodiscard]] std::vector<float> MusicTimesteps(int steps, float shift);

/// APG guidance (apg_forward with dims=[1], eta 0, norm threshold 2.5 and the
/// momentum buffer -0.75) over velocities [frames, 64]: the difference is
/// accumulated with momentum, clamped per channel to norm 2.5 over time,
/// and its component orthogonal to the conditional prediction (per channel,
/// over time) is added (scale - 1) times. The state persists over the
/// steps of one generation.
class ApgGuidance {
public:
  explicit ApgGuidance(float momentum = -0.75F, float norm_threshold = 2.5F)
      : momentum_(momentum), norm_threshold_(norm_threshold) {}
  void Apply(std::span<const float> cond, std::span<const float> uncond,
             float guidance_scale, int frames, std::vector<float>* out);

private:
  float momentum_;
  float norm_threshold_;
  std::vector<double> running_;
};

/// Plain classifier-free guidance: uncond + scale * (cond - uncond).
void CfgGuidance(std::span<const float> cond, std::span<const float> uncond,
                 float guidance_scale, std::vector<float>* out);

/// Latent frame range [start, end) of a repaint span in seconds
/// (_build_chunk_masks_and_src_latents): floor(seconds * 25), start clamped
/// to [0, frames - 1], end to [start + 1, frames].
struct FrameRange {
  int start{0};
  int end{0};
};
[[nodiscard]] FrameRange RepaintFrames(double start_seconds,
                                       double end_seconds, int frames);

/// Context of a repaint / lego task: the chunk mask [frames] is 1 inside the
/// range and 0 outside ("explicit" mode of the official UI), the repaint
/// mask marks the frames to generate, and, unless `lego`, the source
/// latents [frames, 64] inside the range are replaced by the silence latent
/// [frames, 64].
void BuildRepaintContext(const FrameRange& range, int frames, bool lego,
                         std::span<const float> silence,
                         std::vector<float>* src_latents,
                         std::vector<float>* chunk_mask,
                         std::vector<std::uint8_t>* repaint_mask);

/// _repaint_step_injection: outside the mask x_t becomes
/// t_next * noise + (1 - t_next) * clean.
void RepaintInjection(std::span<float> x, std::span<const float> clean,
                      std::span<const std::uint8_t> mask, float t_next,
                      std::span<const float> noise);

/// _repaint_boundary_blend's soft mask [frames]: 1 inside, 0 outside, linear
/// ramps of `crossfade_frames` outside the span (nothing when the mask is
/// all ones or all zeros).
[[nodiscard]] std::vector<float> RepaintBlendWeights(
    std::span<const std::uint8_t> mask, int crossfade_frames);
/// x = w * x + (1 - w) * clean with the weights above.
void RepaintBoundaryBlend(std::span<float> x, std::span<const float> clean,
                          std::span<const std::uint8_t> mask,
                          int crossfade_frames);

/// _resolve_repaint_config: injection ratio, latent crossfade frames and
/// waveform crossfade seconds for "conservative", "balanced" (strength
/// 0..1) or "aggressive".
struct RepaintConfig {
  double injection_ratio{0.5};
  int crossfade_frames{12};
  double wav_crossfade_seconds{0.025};
};
[[nodiscard]] RepaintConfig ResolveRepaintConfig(std::string_view mode,
                                                 double strength);

/// Per-sample weights of the waveform splice: 1 inside [start, end), 0
/// outside, ramps of `crossfade` samples outside the span.
[[nodiscard]] std::vector<float> WaveformCrossfadeMask(std::size_t total,
                                                       std::size_t start,
                                                       std::size_t end,
                                                       std::size_t crossfade);
/// apply_repaint_waveform_splice on planar stereo audio [2][samples]: the
/// source waveform replaces the generated one outside the repaint span
/// (both truncated to the shorter length); nothing happens when the span
/// covers everything.
void RepaintWaveformSplice(std::vector<float>* pred_planar,
                           std::span<const float> src_planar,
                           double start_seconds, double end_seconds,
                           int sample_rate, double crossfade_seconds);

/// Cover noise start: the index of the schedule value (excluding the final
/// 0) nearest to 1 - cover_noise_strength; 0 when the strength is 0.
[[nodiscard]] int CoverNoiseStartIndex(std::span<const float> timesteps,
                                       float cover_noise_strength);

}  // namespace gufo::models::ace_step

#endif  // GUFO_MODELS_ACE_STEP_SAMPLER_HPP_
