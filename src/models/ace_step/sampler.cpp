#include "src/models/ace_step/sampler.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace gufo::models::ace_step {
namespace {

constexpr int kLatents = 64;

std::string Lower(std::string_view text) {
  std::string out(text);
  for (char& c : out)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

std::string Upper(std::string_view text) {
  std::string out(text);
  for (char& c : out)
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return out;
}

// torch.linspace(a, b, n + 2)[1:-1]: n interior points.
std::vector<float> Interior(double a, double b, std::size_t n) {
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i)
    out[i] = static_cast<float>(a + (b - a) * double(i + 1) / double(n + 1));
  return out;
}

}  // namespace

std::string_view MusicTaskName(MusicTask task) {
  switch (task) {
    case MusicTask::kText2Music:
      return "text2music";
    case MusicTask::kRepaint:
      return "repaint";
    case MusicTask::kCover:
      return "cover";
    case MusicTask::kCoverNoFsq:
      return "cover-nofsq";
    case MusicTask::kExtract:
      return "extract";
    case MusicTask::kLego:
      return "lego";
    case MusicTask::kComplete:
      return "complete";
  }
  return "text2music";
}

bool ParseMusicTask(std::string_view name, MusicTask* task) {
  const std::string key = Lower(name);
  if (key.empty() || key == "text2music") {
    *task = MusicTask::kText2Music;
  } else if (key == "repaint") {
    *task = MusicTask::kRepaint;
  } else if (key == "cover") {
    *task = MusicTask::kCover;
  } else if (key == "cover-nofsq" || key == "cover_nofsq") {
    *task = MusicTask::kCoverNoFsq;
  } else if (key == "extract") {
    *task = MusicTask::kExtract;
  } else if (key == "lego") {
    *task = MusicTask::kLego;
  } else if (key == "complete") {
    *task = MusicTask::kComplete;
  } else {
    return false;
  }
  return true;
}

std::string_view DitVariantName(DitVariant variant) {
  switch (variant) {
    case DitVariant::kTurbo:
      return "turbo";
    case DitVariant::kSft:
      return "sft";
    case DitVariant::kBase:
      return "base";
  }
  return "turbo";
}

bool ParseDitVariant(std::string_view name, DitVariant* variant) {
  const std::string key = Lower(name);
  if (key == "turbo")
    *variant = DitVariant::kTurbo;
  else if (key == "sft")
    *variant = DitVariant::kSft;
  else if (key == "base")
    *variant = DitVariant::kBase;
  else
    return false;
  return true;
}

bool GuessDitVariant(std::string_view text, DitVariant* variant) {
  const std::string key = Lower(text);
  if (key.find("turbo") != std::string::npos)
    *variant = DitVariant::kTurbo;
  else if (key.find("sft") != std::string::npos)
    *variant = DitVariant::kSft;
  else if (key.find("base") != std::string::npos)
    *variant = DitVariant::kBase;
  else
    return false;
  return true;
}

SamplerDefaults DefaultSampler(DitVariant variant) {
  switch (variant) {
    case DitVariant::kTurbo:
      return {8, 3.0F, 1.0F};
    case DitVariant::kSft:
      return {50, 3.0F, 7.0F};
    case DitVariant::kBase:
      return {32, 3.0F, 7.0F};
  }
  return {};
}

std::string TaskUnsupportedReason(MusicTask task, DitVariant variant) {
  if (variant == DitVariant::kBase)
    return {};
  switch (task) {
    case MusicTask::kExtract:
    case MusicTask::kLego:
    case MusicTask::kComplete:
      return "task " + std::string(MusicTaskName(task)) +
             " needs the ACE-Step 1.5 XL base model; the loaded " +
             std::string(DitVariantName(variant)) +
             " model supports text2music, repaint, cover and cover-nofsq";
    default:
      return {};
  }
}

bool TaskNeedsSource(MusicTask task) {
  return task != MusicTask::kText2Music;
}

bool TaskIsCover(MusicTask task) {
  return task == MusicTask::kCover;
}

bool TaskRepaints(MusicTask task) {
  return task == MusicTask::kRepaint || task == MusicTask::kLego;
}

std::string DefaultMusicInstruction() {
  return "Fill the audio semantic mask based on the given conditions:";
}

std::string MusicInstruction(MusicTask task, std::string_view track,
                             std::span<const std::string> track_classes) {
  switch (task) {
    case MusicTask::kText2Music:
      return DefaultMusicInstruction();
    case MusicTask::kRepaint:
      return "Repaint the mask area based on the given conditions:";
    case MusicTask::kCover:
    case MusicTask::kCoverNoFsq:
      return "Generate audio semantic tokens based on the given conditions:";
    case MusicTask::kExtract:
      return track.empty() ? "Extract the track from the audio:"
                           : "Extract the " + Upper(track) +
                                 " track from the audio:";
    case MusicTask::kLego:
      return track.empty()
                 ? "Generate the track based on the audio context:"
                 : "Generate the " + Upper(track) +
                       " track based on the audio context:";
    case MusicTask::kComplete: {
      if (track_classes.empty())
        return "Complete the input track:";
      std::string joined;
      for (const std::string& name : track_classes) {
        if (!joined.empty())
          joined += " | ";
        joined += Upper(name);
      }
      return "Complete the input track with " + joined + ":";
    }
  }
  return DefaultMusicInstruction();
}

std::vector<float> MusicTimesteps(int steps, float shift) {
  steps = std::max(steps, 1);
  std::vector<float> t(std::size_t(steps) + 1);
  for (int i = 0; i <= steps; ++i) {
    // torch.linspace(1, 0, steps + 1) in float32.
    float value = static_cast<float>(1.0 - double(i) / double(steps));
    if (i == steps)
      value = 0.0F;
    if (shift != 1.0F)
      value = shift * value / (1.0F + (shift - 1.0F) * value);
    t[std::size_t(i)] = value;
  }
  return t;
}

void ApgGuidance::Apply(std::span<const float> cond,
                        std::span<const float> uncond, float guidance_scale,
                        int frames, std::vector<float>* out) {
  const std::size_t count = std::size_t(frames) * kLatents;
  out->resize(count);
  if (cond.size() != count || uncond.size() != count) {
    std::copy(cond.begin(), cond.end(), out->begin());
    return;
  }
  if (running_.size() != count)
    running_.assign(count, 0.0);
  // diff with momentum: running = diff + momentum * running.
  std::vector<double> diff(count);
  for (std::size_t i = 0; i < count; ++i) {
    running_[i] = double(cond[i]) - double(uncond[i]) +
                  double(momentum_) * running_[i];
    diff[i] = running_[i];
  }
  // Per channel over time: norm clamp, then the orthogonal projection.
  for (int c = 0; c < kLatents; ++c) {
    double norm = 0;
    for (int f = 0; f < frames; ++f)
      norm += diff[std::size_t(f) * kLatents + c] *
              diff[std::size_t(f) * kLatents + c];
    norm = std::sqrt(norm);
    if (norm_threshold_ > 0 && norm > 0) {
      const double scale = std::min(1.0, double(norm_threshold_) / norm);
      for (int f = 0; f < frames; ++f)
        diff[std::size_t(f) * kLatents + c] *= scale;
    }
    // F.normalize(pred_cond, dim=1): v1 = cond / max(||cond||, 1e-12).
    double cond_norm = 0;
    for (int f = 0; f < frames; ++f)
      cond_norm += double(cond[std::size_t(f) * kLatents + c]) *
                   double(cond[std::size_t(f) * kLatents + c]);
    cond_norm = std::max(std::sqrt(cond_norm), 1e-12);
    double dot = 0;
    for (int f = 0; f < frames; ++f)
      dot += diff[std::size_t(f) * kLatents + c] *
             (double(cond[std::size_t(f) * kLatents + c]) / cond_norm);
    for (int f = 0; f < frames; ++f) {
      const std::size_t i = std::size_t(f) * kLatents + c;
      const double v1 = double(cond[i]) / cond_norm;
      const double parallel = dot * v1;
      const double orthogonal = diff[i] - parallel;
      (*out)[i] = static_cast<float>(double(cond[i]) +
                                     (double(guidance_scale) - 1.0) *
                                         orthogonal);
    }
  }
}

void CfgGuidance(std::span<const float> cond, std::span<const float> uncond,
                 float guidance_scale, std::vector<float>* out) {
  out->resize(cond.size());
  for (std::size_t i = 0; i < cond.size(); ++i)
    (*out)[i] = uncond[i] + guidance_scale * (cond[i] - uncond[i]);
}

FrameRange RepaintFrames(double start_seconds, double end_seconds,
                         int frames) {
  FrameRange range;
  // int(seconds * 48000 // 1920) == floor(seconds * 25).
  int start = static_cast<int>(std::floor(start_seconds * 25.0));
  int end = static_cast<int>(std::floor(end_seconds * 25.0));
  start = std::max(0, std::min(start, frames - 1));
  end = std::max(start + 1, std::min(end, frames));
  range.start = start;
  range.end = end;
  return range;
}

void BuildRepaintContext(const FrameRange& range, int frames, bool lego,
                         std::span<const float> silence,
                         std::vector<float>* src_latents,
                         std::vector<float>* chunk_mask,
                         std::vector<std::uint8_t>* repaint_mask) {
  chunk_mask->assign(std::size_t(frames), 0.0F);
  repaint_mask->assign(std::size_t(frames), 0);
  for (int f = range.start; f < range.end && f < frames; ++f) {
    (*chunk_mask)[std::size_t(f)] = 1.0F;
    (*repaint_mask)[std::size_t(f)] = 1;
  }
  if (lego)
    return;
  for (int f = range.start; f < range.end && f < frames; ++f)
    for (int c = 0; c < kLatents; ++c) {
      const std::size_t i = std::size_t(f) * kLatents + c;
      if (i < src_latents->size() && i < silence.size())
        (*src_latents)[i] = silence[i];
    }
}

void RepaintInjection(std::span<float> x, std::span<const float> clean,
                      std::span<const std::uint8_t> mask, float t_next,
                      std::span<const float> noise) {
  for (std::size_t f = 0; f < mask.size(); ++f) {
    if (mask[f])
      continue;
    for (int c = 0; c < kLatents; ++c) {
      const std::size_t i = f * kLatents + c;
      if (i < x.size())
        x[i] = t_next * noise[i] + (1.0F - t_next) * clean[i];
    }
  }
}

std::vector<float> RepaintBlendWeights(std::span<const std::uint8_t> mask,
                                       int crossfade_frames) {
  const std::size_t frames = mask.size();
  std::vector<float> soft(frames);
  for (std::size_t f = 0; f < frames; ++f)
    soft[f] = mask[f] ? 1.0F : 0.0F;
  if (crossfade_frames <= 0)
    return soft;
  std::size_t first = frames, last = 0;
  for (std::size_t f = 0; f < frames; ++f)
    if (mask[f]) {
      first = std::min(first, f);
      last = f;
    }
  if (first == frames || (first == 0 && last + 1 == frames))
    return soft;  // all zeros or all ones
  const std::size_t left = first, right = last + 1;
  const std::size_t fs =
      left > std::size_t(crossfade_frames) ? left - crossfade_frames : 0;
  if (left > fs) {
    const auto ramp = Interior(0.0, 1.0, left - fs);
    std::copy(ramp.begin(), ramp.end(), soft.begin() + std::ptrdiff_t(fs));
  }
  const std::size_t fe = std::min(right + std::size_t(crossfade_frames), frames);
  if (fe > right) {
    const auto ramp = Interior(1.0, 0.0, fe - right);
    std::copy(ramp.begin(), ramp.end(), soft.begin() + std::ptrdiff_t(right));
  }
  return soft;
}

void RepaintBoundaryBlend(std::span<float> x, std::span<const float> clean,
                          std::span<const std::uint8_t> mask,
                          int crossfade_frames) {
  const auto weights = RepaintBlendWeights(mask, crossfade_frames);
  for (std::size_t f = 0; f < weights.size(); ++f)
    for (int c = 0; c < kLatents; ++c) {
      const std::size_t i = f * kLatents + c;
      if (i < x.size())
        x[i] = weights[f] * x[i] + (1.0F - weights[f]) * clean[i];
    }
}

RepaintConfig ResolveRepaintConfig(std::string_view mode, double strength) {
  strength = std::max(0.0, std::min(1.0, strength));
  const std::string key = Lower(mode);
  if (key == "aggressive")
    return {0.0, 0, 0.0};
  if (key == "conservative")
    return {1.0, 25, 0.05};
  const double inv = 1.0 - strength;
  // Python's round() is round-half-even.
  return {inv, static_cast<int>(std::nearbyint(25.0 * inv)), 0.05 * inv};
}

std::vector<float> WaveformCrossfadeMask(std::size_t total, std::size_t start,
                                         std::size_t end,
                                         std::size_t crossfade) {
  std::vector<float> mask(total, 0.0F);
  for (std::size_t i = start; i < end && i < total; ++i)
    mask[i] = 1.0F;
  if (crossfade == 0)
    return mask;
  const std::size_t fade_start = start > crossfade ? start - crossfade : 0;
  if (start > fade_start) {
    const auto ramp = Interior(0.0, 1.0, start - fade_start);
    std::copy(ramp.begin(), ramp.end(),
              mask.begin() + std::ptrdiff_t(fade_start));
  }
  const std::size_t fade_end = std::min(end + crossfade, total);
  if (fade_end > end) {
    const auto ramp = Interior(1.0, 0.0, fade_end - end);
    std::copy(ramp.begin(), ramp.end(), mask.begin() + std::ptrdiff_t(end));
  }
  return mask;
}

void RepaintWaveformSplice(std::vector<float>* pred_planar,
                           std::span<const float> src_planar,
                           double start_seconds, double end_seconds,
                           int sample_rate, double crossfade_seconds) {
  const std::size_t pred_samples = pred_planar->size() / 2;
  const std::size_t src_samples = src_planar.size() / 2;
  const std::size_t samples = std::min(pred_samples, src_samples);
  if (samples == 0)
    return;
  std::size_t start = static_cast<std::size_t>(
      std::max(0.0, std::floor(start_seconds * sample_rate)));
  std::size_t end = static_cast<std::size_t>(
      std::max(0.0, std::floor(end_seconds * sample_rate)));
  start = std::min(start, samples);
  end = std::max(start, std::min(end, samples));
  if (start == 0 && end >= samples)
    return;
  const auto crossfade =
      static_cast<std::size_t>(std::max(0.0, crossfade_seconds * sample_rate));
  const auto mask = WaveformCrossfadeMask(samples, start, end, crossfade);
  for (int ch = 0; ch < 2; ++ch)
    for (std::size_t i = 0; i < samples; ++i) {
      float& p = (*pred_planar)[std::size_t(ch) * pred_samples + i];
      const float s = src_planar[std::size_t(ch) * src_samples + i];
      p = mask[i] * p + (1.0F - mask[i]) * s;
    }
}

int CoverNoiseStartIndex(std::span<const float> timesteps,
                         float cover_noise_strength) {
  if (cover_noise_strength <= 0.0F || timesteps.size() < 2)
    return 0;
  const float level = 1.0F - cover_noise_strength;
  int best = 0;
  float best_distance = std::fabs(timesteps[0] - level);
  for (std::size_t i = 1; i + 1 < timesteps.size(); ++i) {
    const float distance = std::fabs(timesteps[i] - level);
    if (distance < best_distance) {
      best_distance = distance;
      best = static_cast<int>(i);
    }
  }
  return best;
}

}  // namespace gufo::models::ace_step
