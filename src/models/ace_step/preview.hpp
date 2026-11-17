#ifndef GUFO_MODELS_ACE_STEP_PREVIEW_HPP_
#define GUFO_MODELS_ACE_STEP_PREVIEW_HPP_

// Live preview of a running ACE-Step music generation: every few diffusion
// steps the sampler's clean estimate x0 of a short window around the middle
// of the song is decoded with the VAE and handed to a sink as a small mono
// 24 kHz WAV. Everything here is plain CPU code (tested without a GPU); the
// pipeline does the VAE decode.

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "src/models/ace_step/vae.hpp"

namespace gufo::models::ace_step {

/// Latent frames per second of the Oobleck VAE (48000 / 1920 = 25).
constexpr int kLatentFramesPerSecond = kSampleRate / kSamplesPerFrame;
/// Sample rate of the preview WAV (mono 16-bit).
constexpr int kPreviewSampleRate = kSampleRate / 2;
/// Default preview window: about 6 s around the middle of the song.
constexpr double kPreviewWindowSeconds = 6.0;
/// Latent frames decoded in addition on either side of the window and cut
/// from the audio again, so the convolutional VAE does not start cold at
/// the window edges (0.32 s per side).
constexpr int kPreviewMarginFrames = 8;

/// The latent frames of one preview: `first`/`count` is the audible window,
/// `decode_first`/`decode_count` the slice handed to the VAE (window plus
/// margin, clamped to the song). The audible audio starts at sample
/// (first - decode_first) * kSamplesPerFrame of the decoded slice and is
/// count * kSamplesPerFrame samples long (per channel).
struct PreviewWindow {
  int first{0};
  int count{0};
  int decode_first{0};
  int decode_count{0};

  [[nodiscard]] int skip_samples() const noexcept {
    return (first - decode_first) * kSamplesPerFrame;
  }
  [[nodiscard]] int samples() const noexcept {
    return count * kSamplesPerFrame;
  }
  [[nodiscard]] double start_seconds() const noexcept {
    return static_cast<double>(first) / kLatentFramesPerSecond;
  }
  [[nodiscard]] double seconds() const noexcept {
    return static_cast<double>(count) / kLatentFramesPerSecond;
  }
};

/// A `seconds` long window centred on the middle of a song of `frames`
/// latent frames (the whole song when it is shorter), with `margin` extra
/// frames per side for the decode. `frames` <= 0 gives an empty window.
[[nodiscard]] PreviewWindow MusicPreviewWindow(
    int frames, double seconds = kPreviewWindowSeconds,
    int margin = kPreviewMarginFrames);

/// The sampler's clean estimate of the latent at flow time `t`:
///
///   x0 = x_t - t * v
///
/// ACE-Step's rectified flow mixes x_t = (1 - t) * x0 + t * noise and the
/// DiT predicts the velocity v = noise - x0, so the Euler step of the
/// pipeline, x_{t'} = x_t - v * (t - t'), reaches x_t - v * t = x0 at
/// t' = 0: on the last step this estimate *is* the final latent, earlier it
/// is the best guess of the result from the current step. `xt` and `v` have
/// the same length (any slice of the [frames, 64] latent, e.g. one window).
void MusicCleanEstimate(std::span<const float> xt, std::span<const float> v,
                        float t, std::vector<float>* x0);

/// Mono 24 kHz 16-bit PCM WAV of a planar stereo 48 kHz signal
/// [2][samples]: both channels are averaged and each pair of samples is
/// averaged again (2-tap box filter) to halve the rate. A peak above 1 is
/// normalized away first. An odd sample count drops the last sample.
[[nodiscard]] std::string EncodePreviewWav(std::span<const float> planar48k);

/// One published preview.
struct MusicPreview {
  int step{0};   // 1-based diffusion step the estimate comes from
  int steps{0};  // steps of the schedule
  double start_seconds{0};   // window position in the song
  double window_seconds{0};  // window length
  double decode_ms{0};       // VAE decode + WAV encode time
  std::string wav;           // mono 24 kHz 16-bit WAV
};

/// Where the pipeline hands previews. `wanted` is asked before each
/// candidate step (cheap: no decode runs when it is false or missing);
/// `publish` gets the finished preview.
struct MusicPreviewSink {
  std::function<bool()> wanted;
  std::function<void(MusicPreview)> publish;
  /// Preview after every N-th step; the last step is always a candidate.
  int every{2};
  double window_seconds{kPreviewWindowSeconds};
  /// Never decode again while the previous preview is younger than this.
  double min_interval_seconds{3.0};
};

/// Whether step `step` (1-based) of `steps` is a preview candidate with
/// `every`: every N-th step and the last one; `every` < 1 counts as 1.
[[nodiscard]] bool MusicPreviewStep(int step, int steps, int every) noexcept;

}  // namespace gufo::models::ace_step

#endif  // GUFO_MODELS_ACE_STEP_PREVIEW_HPP_
