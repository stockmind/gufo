#ifndef GUFO_MODELS_ACE_STEP_VAE_HPP_
#define GUFO_MODELS_ACE_STEP_VAE_HPP_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::ace_step {

constexpr int kSampleRate = 48000;
constexpr int kSamplesPerFrame = 1920;

/// Decoder of the ACE-Step 1.5 AutoencoderOobleck (Diffusers `vae/` or the
/// ComfyUI `ace_1.5_vae` file): 64 latent channels at 25 frames/s to 48 kHz
/// stereo. Weight norm is folded at load; convolutions run as BF16 GEMMs
/// (im2col) with FP32 activations, Snake in FP32. Long inputs are decoded in
/// overlapping chunks wider than the receptive field, so the result matches
/// a single-pass decode.
class VaeDecoder {
public:
  [[nodiscard]] static std::unique_ptr<VaeDecoder> Load(
      const std::filesystem::path& safetensors, std::string* error);
  ~VaeDecoder();
  VaeDecoder(const VaeDecoder&) = delete;
  VaeDecoder& operator=(const VaeDecoder&) = delete;

  /// latents [frames, 64] -> planar audio [2][frames * 1920].
  bool Decode(std::span<const float> latents, int frames,
              std::vector<float>* audio, std::string* error);

private:
  struct Impl;
  explicit VaeDecoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

/// Encoder of the same Oobleck VAE (the `encoder.*` tensors of the file):
/// 48 kHz stereo to the 128-channel Gaussian parameters (64 means, 64
/// scales) at 25 frames/s, `OobleckDiagonalGaussianDistribution` with
/// std = softplus(scale) + 1e-4. Same numerics as the decoder; long inputs
/// are encoded in overlapping chunks (the official `tiled_encode` trims the
/// overlap the same way).
class VaeEncoder {
public:
  [[nodiscard]] static std::unique_ptr<VaeEncoder> Load(
      const std::filesystem::path& safetensors, std::string* error);
  ~VaeEncoder();
  VaeEncoder(const VaeEncoder&) = delete;
  VaeEncoder& operator=(const VaeEncoder&) = delete;

  /// Planar stereo audio [2][samples] at 48 kHz (samples a multiple of
  /// 1920; a remainder is dropped) -> latents [frames, 64]. With `sample`
  /// the latent is drawn as `latent_dist.sample()` does, with gufo's noise
  /// from `seed`; otherwise it is the mean (`mode()`).
  bool Encode(std::span<const float> planar, std::vector<float>* latents,
              int* frames, bool sample, std::uint64_t seed, std::string* error);

private:
  struct Impl;
  explicit VaeEncoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::ace_step

#endif  // GUFO_MODELS_ACE_STEP_VAE_HPP_
