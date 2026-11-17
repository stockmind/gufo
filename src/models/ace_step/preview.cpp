#include "src/models/ace_step/preview.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace gufo::models::ace_step {
namespace {

void AppendU16(std::string* out, std::uint16_t value) {
  out->push_back(static_cast<char>(value & 0xFFU));
  out->push_back(static_cast<char>(value >> 8U));
}

void AppendU32(std::string* out, std::uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8)
    out->push_back(static_cast<char>((value >> shift) & 0xFFU));
}

}  // namespace

PreviewWindow MusicPreviewWindow(int frames, double seconds, int margin) {
  PreviewWindow w;
  if (frames <= 0)
    return w;
  const int wanted = std::max(
      1, static_cast<int>(std::lround(seconds * kLatentFramesPerSecond)));
  w.count = std::min(wanted, frames);
  w.first = (frames - w.count) / 2;
  margin = std::max(0, margin);
  w.decode_first = std::max(0, w.first - margin);
  const int decode_end = std::min(frames, w.first + w.count + margin);
  w.decode_count = decode_end - w.decode_first;
  return w;
}

void MusicCleanEstimate(std::span<const float> xt, std::span<const float> v,
                        float t, std::vector<float>* x0) {
  const std::size_t n = std::min(xt.size(), v.size());
  x0->resize(n);
  for (std::size_t i = 0; i < n; ++i)
    (*x0)[i] = xt[i] - t * v[i];
}

std::string EncodePreviewWav(std::span<const float> planar48k) {
  const std::size_t samples = planar48k.size() / 2;
  const std::size_t out_samples = samples / 2;
  float peak = 0;
  for (float value : planar48k)
    peak = std::max(peak, std::fabs(value));
  const float gain = peak > 1.0F ? 1.0F / peak : 1.0F;
  const auto data_bytes = static_cast<std::uint32_t>(out_samples * 2);
  std::string out;
  out.reserve(44U + data_bytes);
  out.append("RIFF", 4);
  AppendU32(&out, 36U + data_bytes);
  out.append("WAVEfmt ", 8);
  AppendU32(&out, 16);
  AppendU16(&out, 1);  // PCM
  AppendU16(&out, 1);  // mono
  AppendU32(&out, static_cast<std::uint32_t>(kPreviewSampleRate));
  AppendU32(&out, static_cast<std::uint32_t>(kPreviewSampleRate) * 2U);
  AppendU16(&out, 2);
  AppendU16(&out, 16);
  out.append("data", 4);
  AppendU32(&out, data_bytes);
  const float* left = planar48k.data();
  const float* right = planar48k.data() + samples;
  for (std::size_t i = 0; i < out_samples; ++i) {
    const std::size_t a = 2 * i;
    const float mono =
        (left[a] + right[a] + left[a + 1] + right[a + 1]) * 0.25F * gain;
    const float v = std::clamp(mono, -1.0F, 1.0F);
    AppendU16(&out, static_cast<std::uint16_t>(static_cast<std::int16_t>(
                        std::lround(v * 32767.0F))));
  }
  return out;
}

bool MusicPreviewStep(int step, int steps, int every) noexcept {
  if (step < 1 || steps < 1)
    return false;
  if (step == steps)
    return true;
  return step % std::max(every, 1) == 0;
}

}  // namespace gufo::models::ace_step
