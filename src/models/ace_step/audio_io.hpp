// Audio in and out of the ACE-Step pipeline without a model: WAV decoding
// (PCM 8/16/24/32-bit and float, any channel count), torchaudio-style
// resampling to 48 kHz stereo, the reference-audio excerpt of the official
// pipeline and the 16-bit stereo WAV writer. Written with AI assistance
// (Claude Code).
#ifndef GUFO_MODELS_ACE_STEP_AUDIO_IO_HPP_
#define GUFO_MODELS_ACE_STEP_AUDIO_IO_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::ace_step {

/// Planar stereo audio [2][samples].
struct StereoAudio {
  int sample_rate{0};
  std::size_t samples{0};
  std::vector<float> planar;
};

/// RIFF/WAVE with PCM (8, 16, 24, 32-bit), IEEE float (32, 64-bit) or
/// WAVE_FORMAT_EXTENSIBLE of those. Mono is duplicated, extra channels are
/// dropped (the official `_normalize_audio_to_stereo_48k`). False with
/// `error` for anything else.
[[nodiscard]] bool DecodeWavStereo(std::span<const std::uint8_t> bytes,
                                   StereoAudio* audio, std::string* error);

/// torchaudio.transforms.Resample(from, to) defaults (sinc_interp_hann,
/// lowpass_filter_width 6, rolloff 0.99); identical rates copy.
[[nodiscard]] std::vector<float> ResampleLikeTorchaudio(
    std::span<const float> input, std::int64_t from, std::int64_t to);

/// Resamples both channels to 48 kHz and clamps to [-1, 1].
void NormalizeStereo48k(StereoAudio* audio);

/// True when every sample is below 1e-6 in magnitude (`is_silence`).
[[nodiscard]] bool IsSilentAudio(std::span<const float> planar);

/// `process_reference_audio`: the 48 kHz stereo reference is repeated to at
/// least 30 s, split in three equal parts and a 10 s excerpt is taken from
/// each (the official code picks the offsets at random; here they come from
/// `seed`), giving exactly 30 s = 750 latent frames.
[[nodiscard]] std::vector<float> ReferenceExcerpt(const StereoAudio& audio,
                                                  std::uint64_t seed);

/// 16-bit PCM WAV of planar stereo audio.
[[nodiscard]] std::string EncodeStereoWav(std::span<const float> planar,
                                          int sample_rate);

}  // namespace gufo::models::ace_step

#endif  // GUFO_MODELS_ACE_STEP_AUDIO_IO_HPP_
