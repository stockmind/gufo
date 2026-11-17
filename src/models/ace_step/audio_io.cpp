#include "src/models/ace_step/audio_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>

namespace gufo::models::ace_step {
namespace {

constexpr double kPi = 3.14159265358979323846;

std::uint32_t U32(std::span<const std::uint8_t> b, std::size_t at) {
  return std::uint32_t(b[at]) | (std::uint32_t(b[at + 1]) << 8) |
         (std::uint32_t(b[at + 2]) << 16) | (std::uint32_t(b[at + 3]) << 24);
}

std::uint16_t U16(std::span<const std::uint8_t> b, std::size_t at) {
  return static_cast<std::uint16_t>(b[at] | (b[at + 1] << 8));
}

void AppendU16(std::string* out, std::uint16_t value) {
  out->push_back(static_cast<char>(value & 0xFFU));
  out->push_back(static_cast<char>(value >> 8U));
}

void AppendU32(std::string* out, std::uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8)
    out->push_back(static_cast<char>((value >> shift) & 0xFFU));
}

bool Fail(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
  return false;
}

}  // namespace

bool DecodeWavStereo(std::span<const std::uint8_t> bytes, StereoAudio* audio,
                     std::string* error) {
  if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
      std::memcmp(bytes.data() + 8, "WAVE", 4) != 0)
    return Fail(error, "not a RIFF/WAVE file");
  std::uint16_t format = 0, channels = 0, bits = 0;
  std::uint32_t rate = 0;
  bool have_fmt = false;
  std::size_t data_at = 0, data_size = 0;
  std::size_t at = 12;
  while (at + 8 <= bytes.size()) {
    const std::uint32_t size = U32(bytes, at + 4);
    const std::size_t body = at + 8;
    const std::size_t available = bytes.size() - body;
    const std::size_t length = std::min<std::size_t>(size, available);
    if (std::memcmp(bytes.data() + at, "fmt ", 4) == 0 && length >= 16) {
      format = U16(bytes, body);
      channels = U16(bytes, body + 2);
      rate = U32(bytes, body + 4);
      bits = U16(bytes, body + 14);
      if (format == 0xFFFE && length >= 26)  // WAVE_FORMAT_EXTENSIBLE
        format = U16(bytes, body + 24);
      have_fmt = true;
    } else if (std::memcmp(bytes.data() + at, "data", 4) == 0) {
      data_at = body;
      data_size = length;
      if (have_fmt)
        break;
    }
    at = body + length + (length & 1U);
  }
  if (!have_fmt || data_at == 0)
    return Fail(error, "WAV without fmt or data chunk");
  if (channels == 0 || rate == 0)
    return Fail(error, "WAV without channels or sample rate");
  const bool is_float = format == 3;
  if (!(format == 1 || format == 3))
    return Fail(error, "unsupported WAV format " + std::to_string(format));
  if ((is_float && bits != 32 && bits != 64) ||
      (!is_float && bits != 8 && bits != 16 && bits != 24 && bits != 32))
    return Fail(error, "unsupported WAV bit depth " + std::to_string(bits));
  const std::size_t bytes_per = bits / 8;
  const std::size_t frame = bytes_per * channels;
  const std::size_t samples = data_size / frame;
  if (samples == 0)
    return Fail(error, "WAV without samples");
  audio->sample_rate = static_cast<int>(rate);
  audio->samples = samples;
  audio->planar.assign(2 * samples, 0.0F);
  const int out_channels = std::min<int>(channels, 2);
  for (std::size_t i = 0; i < samples; ++i) {
    for (int ch = 0; ch < out_channels; ++ch) {
      const std::uint8_t* p = bytes.data() + data_at + i * frame + ch * bytes_per;
      float value = 0;
      if (is_float && bits == 32) {
        std::uint32_t u = U32(bytes, std::size_t(p - bytes.data()));
        std::memcpy(&value, &u, 4);
      } else if (is_float) {
        std::uint64_t u = 0;
        for (int k = 7; k >= 0; --k)
          u = (u << 8) | p[k];
        double d = 0;
        std::memcpy(&d, &u, 8);
        value = static_cast<float>(d);
      } else if (bits == 8) {
        value = (float(p[0]) - 128.0F) / 128.0F;
      } else if (bits == 16) {
        value = float(static_cast<std::int16_t>(U16(bytes,
                                                    std::size_t(p - bytes.data())))) /
                32768.0F;
      } else if (bits == 24) {
        std::int32_t v = (p[0] | (p[1] << 8) | (p[2] << 16));
        if (v & 0x800000)
          v |= ~0xFFFFFF;
        value = float(v) / 8388608.0F;
      } else {
        value = float(static_cast<std::int32_t>(
                    U32(bytes, std::size_t(p - bytes.data())))) /
                2147483648.0F;
      }
      audio->planar[std::size_t(ch) * samples + i] = value;
    }
    if (channels == 1)
      audio->planar[samples + i] = audio->planar[i];
  }
  return true;
}

std::vector<float> ResampleLikeTorchaudio(std::span<const float> input,
                                          std::int64_t from, std::int64_t to) {
  if (from == to || input.empty() || from <= 0 || to <= 0)
    return {input.begin(), input.end()};
  const std::int64_t g = std::gcd(from, to);
  const std::int64_t orig = from / g, fresh = to / g;
  constexpr double kRolloff = 0.99;
  constexpr double kWidth = 6.0;  // lowpass_filter_width
  const double base_freq = double(std::min(orig, fresh)) * kRolloff;
  const auto width = static_cast<std::int64_t>(
      std::ceil(kWidth * double(orig) / base_freq));
  const std::int64_t taps = 2 * width + orig;
  std::vector<float> kernels(std::size_t(fresh) * std::size_t(taps));
  for (std::int64_t i = 0; i < fresh; ++i)
    for (std::int64_t j = 0; j < taps; ++j) {
      double t = -double(i) / double(fresh) + double(j - width) / double(orig);
      t *= base_freq;
      t = std::max(-kWidth, std::min(kWidth, t));
      const double window = std::pow(std::cos(t * kPi / kWidth / 2.0), 2.0);
      t *= kPi;
      const double sinc = t == 0.0 ? 1.0 : std::sin(t) / t;
      kernels[std::size_t(i * taps + j)] =
          static_cast<float>(sinc * window * base_freq / double(orig));
    }
  const auto length = static_cast<std::int64_t>(input.size());
  const auto target = static_cast<std::int64_t>(
      std::ceil(double(fresh) * double(length) / double(orig)));
  // conv1d over the input padded by `width` in front and `width + orig`
  // behind, stride `orig`.
  const std::int64_t padded = length + 2 * width + orig;
  const std::int64_t frames = (padded - taps) / orig + 1;
  std::vector<float> out(std::size_t(frames) * std::size_t(fresh));
  for (std::int64_t f = 0; f < frames; ++f)
    for (std::int64_t i = 0; i < fresh; ++i) {
      double sum = 0;
      const float* kernel = kernels.data() + std::size_t(i * taps);
      for (std::int64_t j = 0; j < taps; ++j) {
        const std::int64_t n = f * orig + j - width;
        if (n >= 0 && n < length)
          sum += double(kernel[j]) * double(input[std::size_t(n)]);
      }
      out[std::size_t(f * fresh + i)] = static_cast<float>(sum);
    }
  out.resize(std::size_t(std::min(target, frames * fresh)));
  return out;
}

void NormalizeStereo48k(StereoAudio* audio) {
  constexpr int kRate = 48000;
  if (audio->sample_rate != kRate && audio->samples > 0) {
    std::vector<float> planar;
    std::size_t samples = 0;
    for (int ch = 0; ch < 2; ++ch) {
      auto channel = ResampleLikeTorchaudio(
          std::span<const float>(audio->planar)
              .subspan(std::size_t(ch) * audio->samples, audio->samples),
          audio->sample_rate, kRate);
      if (ch == 0) {
        samples = channel.size();
        planar.assign(2 * samples, 0.0F);
      }
      std::copy_n(channel.begin(), std::min(samples, channel.size()),
                  planar.begin() + std::ptrdiff_t(std::size_t(ch) * samples));
    }
    audio->planar = std::move(planar);
    audio->samples = samples;
    audio->sample_rate = kRate;
  }
  for (float& v : audio->planar)
    v = std::clamp(v, -1.0F, 1.0F);
}

bool IsSilentAudio(std::span<const float> planar) {
  return std::all_of(planar.begin(), planar.end(),
                     [](float v) { return std::fabs(v) < 1e-6F; });
}

std::vector<float> ReferenceExcerpt(const StereoAudio& audio,
                                    std::uint64_t seed) {
  constexpr std::size_t kTarget = 30 * 48000, kSegment = 10 * 48000;
  const std::size_t n = audio.samples;
  std::vector<float> out(2 * 3 * kSegment, 0.0F);
  if (n == 0)
    return out;
  // Repeat to at least 30 s.
  const std::size_t repeats = (kTarget + n - 1) / n;
  const std::size_t total = n * repeats;
  auto sample = [&](int ch, std::size_t i) {
    return audio.planar[std::size_t(ch) * n + (i % n)];
  };
  std::mt19937_64 rng(seed);
  auto draw = [&](std::size_t upper) -> std::size_t {
    return upper == 0 ? 0
                      : std::uniform_int_distribution<std::size_t>(0, upper)(rng);
  };
  const std::size_t part = total / 3;
  const std::size_t starts[3] = {
      draw(part > kSegment ? part - kSegment : 0),
      part + draw(part > kSegment ? part - kSegment : 0),
      2 * part +
          draw(total - 2 * part > kSegment ? total - 2 * part - kSegment : 0)};
  for (int s = 0; s < 3; ++s)
    for (std::size_t i = 0; i < kSegment; ++i) {
      const std::size_t src = starts[s] + i;
      if (src >= total)
        break;
      for (int ch = 0; ch < 2; ++ch)
        out[std::size_t(ch) * 3 * kSegment + std::size_t(s) * kSegment + i] =
            sample(ch, src);
    }
  return out;
}

std::string EncodeStereoWav(std::span<const float> planar, int sample_rate) {
  const std::size_t samples = planar.size() / 2;
  const auto data_bytes = static_cast<std::uint32_t>(samples * 4);
  std::string out;
  out.reserve(44U + data_bytes);
  out.append("RIFF", 4);
  AppendU32(&out, 36U + data_bytes);
  out.append("WAVEfmt ", 8);
  AppendU32(&out, 16);
  AppendU16(&out, 1);  // PCM
  AppendU16(&out, 2);
  AppendU32(&out, static_cast<std::uint32_t>(sample_rate));
  AppendU32(&out, static_cast<std::uint32_t>(sample_rate) * 4U);
  AppendU16(&out, 4);
  AppendU16(&out, 16);
  out.append("data", 4);
  AppendU32(&out, data_bytes);
  for (std::size_t i = 0; i < samples; ++i)
    for (int ch = 0; ch < 2; ++ch) {
      const float v = std::clamp(planar[ch * samples + i], -1.0F, 1.0F);
      AppendU16(&out, static_cast<std::uint16_t>(static_cast<std::int16_t>(
                          std::lround(v * 32767.0F))));
    }
  return out;
}

}  // namespace gufo::models::ace_step
