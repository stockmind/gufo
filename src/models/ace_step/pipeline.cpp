#include "src/models/ace_step/pipeline.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>

#include "src/models/ace_step/model.hpp"
#include "src/models/ace_step/tokenizer.hpp"
#include "src/models/ace_step/vae.hpp"
#include "src/models/ace_step/weights.hpp"

namespace gufo::models::ace_step {
namespace {

constexpr int kMaxTextTokens = 256;
constexpr int kMaxLyricTokens = 2048;
constexpr std::uint32_t kEndOfText = 151643;
constexpr double kMaxSeconds = 600.0;
constexpr int kMaxSteps = 200;

double Since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

std::string Lower(std::string_view text) {
  std::string out(text);
  for (char& c : out)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

}  // namespace

// The host side (paths, prompts, noise, WAV, file resolution) is in prompt.cpp.

struct MusicPipeline::Impl {
  std::unique_ptr<Tokenizer> tokenizer;
  std::unique_ptr<TextEncoder> text;
  std::unique_ptr<Dit> dit;
  std::unique_ptr<VaeDecoder> vae;
  std::unique_ptr<VaeEncoder> encoder;  // loaded on the first source task
  std::filesystem::path lm_path, dit_path, vae_path;
  DitVariant variant{DitVariant::kTurbo};
  std::vector<float> silence;  // [frames, 64]
  int silence_frames{0};

  // Silence latent frames [0, frames), repeating the tail when longer.
  std::vector<float> Silence(int frames) const {
    std::vector<float> out(std::size_t(frames) * kLatentChannels);
    for (int f = 0; f < frames; ++f) {
      const int s = std::min(f, silence_frames - 1);
      std::copy_n(silence.begin() + std::size_t(s) * kLatentChannels,
                  kLatentChannels,
                  out.begin() + std::size_t(f) * kLatentChannels);
    }
    return out;
  }

  bool EnsureEncoder(std::string* error) {
    if (encoder)
      return true;
    encoder = VaeEncoder::Load(vae_path, error);
    return encoder != nullptr;
  }

  // Tokenizes a prompt with the <|endoftext|> the Qwen3-Embedding tokenizer
  // appends after truncation.
  bool Encode(const std::string& prompt, int limit,
              std::vector<std::uint32_t>* ids, std::string* error) {
    try {
      *ids = tokenizer->Encode(prompt);
    } catch (const std::exception& e) {
      return Fail(error, std::string("ACE-Step tokenizer: ") + e.what());
    }
    if (ids->size() > std::size_t(limit - 1))
      ids->resize(std::size_t(limit - 1));
    ids->push_back(kEndOfText);
    return true;
  }
};

MusicPipeline::MusicPipeline(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
MusicPipeline::~MusicPipeline() = default;

DitVariant MusicPipeline::variant() const {
  return impl_->variant;
}
bool MusicPipeline::has_lm() const {
  return !impl_->lm_path.empty();
}

std::unique_ptr<MusicPipeline> MusicPipeline::Load(const ModelPaths& paths,
                                                   std::string* error) {
  auto impl = std::make_unique<Impl>();
  try {
    impl->tokenizer = std::make_unique<Tokenizer>(paths.tokenizer);
  } catch (const std::exception& e) {
    Fail(error, std::string("ACE-Step tokenizer: ") + e.what());
    return nullptr;
  }
  {
    SafetensorsFile file;
    const auto* tensor = file.Open(paths.silence, nullptr, error)
                             ? file.Find("silence_latent")
                             : nullptr;
    if (tensor == nullptr || tensor->shape.size() != 2 ||
        tensor->shape[1] != kLatentChannels ||
        tensor->shape[0] < std::size_t(kTimbreFrames) ||
        !file.ReadHost("silence_latent", tensor->count(), &impl->silence,
                       error)) {
      if (error != nullptr && error->empty())
        *error = "invalid ACE-Step silence latent " + paths.silence.string();
      return nullptr;
    }
    impl->silence_frames = static_cast<int>(tensor->shape[0]);
  }
  impl->text = TextEncoder::Load(paths.text, error);
  if (!impl->text)
    return nullptr;
  impl->dit = Dit::Load(paths.dit, error);
  if (!impl->dit)
    return nullptr;
  impl->vae = VaeDecoder::Load(paths.vae, error);
  if (!impl->vae)
    return nullptr;
  // The LM is loaded per request and released before the DiT runs, so
  // only one large model holds device memory at a time.
  impl->lm_path = paths.lm;
  impl->dit_path = paths.dit;
  impl->vae_path = paths.vae;
  impl->variant = paths.variant;
  return std::unique_ptr<MusicPipeline>(new MusicPipeline(std::move(impl)));
}

namespace {

// Decodes the preview window of the clean estimate x0 = x - t * v (see
// MusicCleanEstimate) and hands the mono WAV to the sink. A failed decode
// only drops this preview; the generation itself goes on.
void PublishPreview(VaeDecoder& vae, const MusicPreviewSink& sink,
                    std::span<const float> x, std::span<const float> v,
                    float t, int frames, int step, int steps) {
  const auto start = std::chrono::steady_clock::now();
  const PreviewWindow window =
      MusicPreviewWindow(frames, sink.window_seconds, kPreviewMarginFrames);
  if (window.decode_count <= 0)
    return;
  const std::size_t offset =
      std::size_t(window.decode_first) * kLatentChannels;
  const std::size_t length =
      std::size_t(window.decode_count) * kLatentChannels;
  std::vector<float> x0;
  MusicCleanEstimate(x.subspan(offset, length), v.subspan(offset, length), t,
                     &x0);
  std::vector<float> audio;
  std::string error;
  if (!vae.Decode(x0, window.decode_count, &audio, &error))
    return;
  // Cut the margin: the decoded slice is planar [2][decode_count * 1920].
  const std::size_t decoded = audio.size() / 2;
  const auto skip = std::size_t(window.skip_samples());
  const auto keep = std::size_t(window.samples());
  if (skip + keep > decoded)
    return;
  std::vector<float> planar(2 * keep);
  std::copy_n(audio.begin() + skip, keep, planar.begin());
  std::copy_n(audio.begin() + decoded + skip, keep, planar.begin() + keep);
  MusicPreview preview;
  preview.step = step;
  preview.steps = steps;
  preview.start_seconds = window.start_seconds();
  preview.window_seconds = window.seconds();
  preview.wav = EncodePreviewWav(planar);
  preview.decode_ms = Since(start) * 1000.0;
  if (sink.publish)
    sink.publish(std::move(preview));
}

// Planar stereo audio padded with `left` / `right` zero samples per channel.
std::vector<float> PadPlanar(std::span<const float> planar, std::size_t left,
                             std::size_t right) {
  const std::size_t samples = planar.size() / 2;
  const std::size_t total = left + samples + right;
  std::vector<float> out(2 * total, 0.0F);
  for (int ch = 0; ch < 2; ++ch)
    std::copy_n(planar.begin() + std::ptrdiff_t(std::size_t(ch) * samples),
                samples,
                out.begin() + std::ptrdiff_t(std::size_t(ch) * total + left));
  return out;
}

// Context latents [frames, 128]: source latents | chunk mask (per frame).
std::vector<float> MakeContext(std::span<const float> source,
                               std::span<const float> chunk_mask, int frames) {
  std::vector<float> context(std::size_t(frames) * 2 * kLatentChannels);
  for (int f = 0; f < frames; ++f) {
    std::copy_n(source.begin() + std::size_t(f) * kLatentChannels,
                kLatentChannels,
                context.begin() + std::size_t(f) * 2 * kLatentChannels);
    std::fill_n(context.begin() + std::size_t(f) * 2 * kLatentChannels +
                    kLatentChannels,
                kLatentChannels, chunk_mask[std::size_t(f)]);
  }
  return context;
}

// Diagnostics: GUFO_ACE_DUMP=<dir> writes the raw source latents, the FSQ
// codes and the detokenized hints of a request as flat little-endian files.
template <typename T>
void DumpRaw(const char* dir, const char* name, const std::vector<T>& data) {
  std::ofstream out(std::filesystem::path(dir) / name, std::ios::binary);
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(T)));
}

}  // namespace

bool MusicPipeline::Generate(const MusicRequest& request,
                             std::vector<float>* audio,
                             MusicTelemetry* telemetry,
                             const MusicProgress& progress,
                             std::string* error,
                             const MusicPreviewSink* preview) {
  Impl& m = *impl_;
  MusicTelemetry local;
  MusicTelemetry& tm = telemetry != nullptr ? *telemetry : local;
  const MusicTask task = request.task;
  const SamplerDefaults defaults = DefaultSampler(m.variant);
  const int steps = request.steps > 0 ? request.steps : defaults.steps;
  const float shift = request.shift > 0.0F ? request.shift : defaults.shift;
  // Turbo bakes its guidance into the distillation: the official pipeline
  // forces guidance 1 there.
  const float guidance =
      m.variant == DitVariant::kTurbo
          ? 1.0F
          : (request.guidance_scale > 0.0F ? request.guidance_scale
                                           : defaults.guidance_scale);
  if (steps < 1 || steps > kMaxSteps)
    return Fail(error, "steps must be 1.." + std::to_string(kMaxSteps));
  if (const std::string reason = TaskUnsupportedReason(task, m.variant);
      !reason.empty())
    return Fail(error, reason);
  const bool has_source = !request.source_audio.empty();
  if (TaskNeedsSource(task) && !has_source)
    return Fail(error, "task " + std::string(MusicTaskName(task)) +
                           " requires source_audio");
  if (!TaskNeedsSource(task) &&
      !(request.duration >= 1.0 && request.duration <= kMaxSeconds))
    return Fail(error, "duration must be 1..600 seconds");
  auto report = [&](double fraction, std::string_view phase) {
    return !progress || progress(fraction, phase);
  };

  // Source audio -> latents. Repaint / lego may extend the source in front
  // (negative start) or behind (end beyond the source): the audio is zero
  // padded and the span moves by the front padding, as the official
  // prepare_padding_info does. The duration of every source task is the
  // (padded) source length; a remainder below one latent frame is dropped.
  auto start = std::chrono::steady_clock::now();
  std::vector<float> source_audio, source_latents, clean_latents;
  double repaint_start = request.repaint_start;
  double repaint_end = request.repaint_end;
  int frames = 0;
  double duration = request.duration;
  if (TaskNeedsSource(task)) {
    std::size_t samples = request.source_audio.size() / 2;
    if (samples < std::size_t(kSamplesPerFrame))
      return Fail(error, "source_audio is shorter than one latent frame");
    std::size_t left = 0, right = 0;
    if (TaskRepaints(task)) {
      const double source_seconds = double(samples) / kSampleRate;
      const double actual_end =
          repaint_end < 0.0 ? source_seconds : repaint_end;
      const double left_seconds = std::max(0.0, -repaint_start);
      const double right_seconds = std::max(0.0, actual_end - source_seconds);
      if (actual_end <= repaint_start)
        return Fail(error, "repaint_end must be greater than repaint_start");
      if (actual_end + left_seconds > kMaxSeconds)
        return Fail(error, "repaint span exceeds 600 seconds");
      left = static_cast<std::size_t>(left_seconds * kSampleRate);
      right = static_cast<std::size_t>(right_seconds * kSampleRate);
      repaint_start += left_seconds;
      repaint_end = actual_end + left_seconds;
    }
    source_audio = PadPlanar(request.source_audio, left, right);
    samples = source_audio.size() / 2;
    frames = static_cast<int>(samples / kSamplesPerFrame);
    if (frames < 1)
      return Fail(error, "source_audio is shorter than one latent frame");
    if (frames > static_cast<int>(kMaxSeconds * kLatentFramesPerSecond))
      return Fail(error, "source_audio is longer than 600 seconds");
    if (!report(0.01, "encode"))
      return Fail(error, "cancelled");
    if (!m.EnsureEncoder(error))
      return false;
    int encoded = 0;
    if (!m.encoder->Encode(source_audio, &source_latents, &encoded,
                           request.vae_sample, request.seed ^ 0x5ACE5EEDULL,
                           error))
      return false;
    frames = encoded;
    duration = double(frames) / kLatentFramesPerSecond;
    clean_latents = source_latents;
  } else {
    frames = static_cast<int>(
        std::lround(request.duration * kSampleRate / kSamplesPerFrame));
  }
  tm.frames = frames;
  tm.steps = steps;
  // Timbre reference: the official 3 x 10 s excerpt, VAE-encoded (750
  // frames); silence without a reference.
  std::vector<float> timbre;
  if (!request.reference_audio.empty()) {
    StereoAudio reference;
    reference.sample_rate = kSampleRate;
    reference.samples = request.reference_audio.size() / 2;
    reference.planar = request.reference_audio;
    if (reference.samples == 0 || IsSilentAudio(reference.planar))
      return Fail(error, "reference_audio is empty or silent");
    const auto excerpt = ReferenceExcerpt(reference, request.seed);
    int reference_frames = 0;
    if (!m.EnsureEncoder(error) ||
        !m.encoder->Encode(excerpt, &timbre, &reference_frames,
                           request.vae_sample, request.seed ^ 0x7E4B2EULL,
                           error))
      return false;
    if (reference_frames != kTimbreFrames)
      return Fail(error, "reference excerpt did not encode to 750 frames");
  } else {
    timbre = m.Silence(kTimbreFrames);
  }
  tm.encode_seconds = Since(start);

  start = std::chrono::steady_clock::now();
  std::string instruction =
      MusicInstruction(task, request.track, request.track_classes);
  // Diagnostics: GUFO_ACE_INSTRUCTION overrides the DiT task instruction.
  if (const char* override_instruction = std::getenv("GUFO_ACE_INSTRUCTION");
      override_instruction != nullptr && *override_instruction != '\0')
    instruction = override_instruction;
  std::vector<std::uint32_t> text_ids, lyric_ids;
  if (!m.Encode(MusicTextPrompt(request, instruction, duration),
                kMaxTextTokens, &text_ids, error) ||
      !m.Encode(MusicLyricPrompt(request), kMaxLyricTokens, &lyric_ids, error))
    return false;
  tm.text_tokens = static_cast<int>(text_ids.size());
  tm.lyric_tokens = static_cast<int>(lyric_ids.size());
  std::vector<float> text, lyric, conditions;
  if (!m.text->Encode(text_ids, &text, error) ||
      !m.text->Embed(lyric_ids, &lyric, error))
    return false;
  // The non-cover branch of audio_cover_strength < 1: default instruction,
  // silence context.
  const float cover_strength =
      task == MusicTask::kText2Music
          ? 1.0F
          : std::clamp(request.cover_strength, 0.0F, 1.0F);
  // The plain prompt carries a different instruction, so its token count
  // differs from the task prompt's: it must travel with its own row count.
  std::vector<float> text_plain;
  int plain_tokens = 0;
  if (cover_strength < 1.0F) {
    std::vector<std::uint32_t> plain_ids;
    if (!m.Encode(MusicTextPrompt(request, DefaultMusicInstruction(), duration),
                  kMaxTextTokens, &plain_ids, error) ||
        !m.text->Encode(plain_ids, &text_plain, error))
      return false;
    plain_tokens = static_cast<int>(plain_ids.size());
  }
  tm.text_seconds = Since(start);
  if (!report(0.05, "conditions"))
    return Fail(error, "cancelled");

  // Source latents of the context: the 5 Hz LM's planned audio codes
  // through the detokenizer (text2music with the LM), the encoded source
  // (source tasks) or silence; all-ones chunk mask unless repainting. The
  // LM and the DiT never hold device memory at the same time.
  std::vector<int> codes;
  const bool plan = task == MusicTask::kText2Music && !m.lm_path.empty() &&
                    request.audio_codes;
  if (plan) {
    const auto lm_start = std::chrono::steady_clock::now();
    m.dit.reset();
    auto lm = CodeLm::Load(m.lm_path, error);
    if (!lm)
      return false;
    std::vector<std::uint32_t> cond, uncond;
    try {
      cond = m.tokenizer->Encode(MusicLmPrompt(request));
      uncond = m.tokenizer->Encode(MusicLmUnconditionalPrompt());
    } catch (const std::exception& e) {
      return Fail(error, std::string("ACE-Step tokenizer: ") + e.what());
    }
    CodeSampling sampling;
    sampling.cfg = request.lm_cfg;
    sampling.temperature = request.lm_temperature;
    sampling.top_p = request.lm_top_p;
    sampling.seed = request.seed;
    const int count = static_cast<int>(request.duration * 5);
    if (!lm->Generate(
            cond, uncond, count, sampling, &codes,
            [&](int done) {
              return report(0.05 + 0.4 * done / std::max(count, 1), "codes");
            },
            error))
      return false;
    lm.reset();
    tm.codes = count;
    tm.lm_seconds = Since(lm_start);
  }
  start = std::chrono::steady_clock::now();
  if (m.dit == nullptr) {
    m.dit = Dit::Load(m.dit_path, error);
    if (!m.dit)
      return false;
  }
  std::vector<float> source;
  if (plan) {
    if (!m.dit->CodeHints(codes, frames, &source, error))
      return false;
  } else if (has_source) {
    source = source_latents;
  } else {
    source = m.Silence(frames);
  }
  std::vector<float> chunk_mask(std::size_t(frames), 1.0F);
  std::vector<std::uint8_t> repaint_mask;
  if (TaskRepaints(task)) {
    const FrameRange range = RepaintFrames(repaint_start, repaint_end, frames);
    BuildRepaintContext(range, frames, task == MusicTask::kLego,
                        m.Silence(frames), &source, &chunk_mask,
                        &repaint_mask);
  }
  // The renoised cover start uses the source before the FSQ round trip.
  const std::vector<float> source_pre_fsq = source;
  if (TaskIsCover(task)) {
    if (!m.dit->Tokenize(source, frames, m.Silence(5), &codes, error) ||
        !m.dit->CodeHints(codes, frames, &source, error))
      return false;
    tm.codes = static_cast<int>(codes.size());
  }
  if (const char* dump = std::getenv("GUFO_ACE_DUMP");
      dump != nullptr && *dump != '\0') {
    DumpRaw(dump, "source_latents.f32", source_pre_fsq);
    DumpRaw(dump, "codes.i32", codes);
    DumpRaw(dump, "hints.f32", source);
  }
  // Diagnostics: GUFO_ACE_HINTS_FILE=<file> replaces the cover hints with
  // [frames, 64] float32 read from the file (reference hints for A/B tests).
  if (const char* inject = std::getenv("GUFO_ACE_HINTS_FILE");
      inject != nullptr && *inject != '\0' && TaskIsCover(task)) {
    std::ifstream in(inject, std::ios::binary);
    std::vector<float> file(std::size_t(frames) * kLatentChannels);
    in.read(reinterpret_cast<char*>(file.data()),
            static_cast<std::streamsize>(file.size() * sizeof(float)));
    if (in.gcount() == static_cast<std::streamsize>(file.size() * sizeof(float)))
      source = file;
    else
      return Fail(error, "GUFO_ACE_HINTS_FILE has the wrong size");
  }
  if (!m.dit->EncodeConditions(text, tm.text_tokens, lyric, tm.lyric_tokens,
                               timbre, kTimbreFrames, &conditions, error))
    return false;
  const int rows = static_cast<int>(conditions.size() / kConditionWidth);
  if (!m.dit->SetConditions(conditions, rows,
                            MakeContext(source, chunk_mask, frames), frames,
                            error))
    return false;
  std::vector<float> conditions_plain, context_plain;
  if (cover_strength < 1.0F) {
    if (!m.dit->EncodeConditions(text_plain, plain_tokens, lyric,
                                 tm.lyric_tokens, timbre, kTimbreFrames,
                                 &conditions_plain, error))
      return false;
    context_plain = MakeContext(m.Silence(frames), chunk_mask, frames);
  }
  tm.condition_seconds = Since(start);

  // Schedule, noise and the optional renoised cover start.
  start = std::chrono::steady_clock::now();
  std::vector<float> times = MusicTimesteps(steps, shift);
  const auto noise = MusicNoise(request.seed, std::size_t(frames) * kLatentChannels);
  std::vector<float> x = noise;
  const float cover_noise =
      task == MusicTask::kText2Music
          ? 0.0F
          : std::clamp(request.cover_noise_strength, 0.0F, 1.0F);
  if (cover_noise > 0.0F) {
    const int first = CoverNoiseStartIndex(times, cover_noise);
    const float t0 = times[std::size_t(first)];
    for (std::size_t i = 0; i < x.size(); ++i)
      x[i] = t0 * noise[i] + (1.0F - t0) * source_pre_fsq[i];
    times.erase(times.begin(), times.begin() + first);
  }
  const int infer_steps = static_cast<int>(times.size()) - 1;
  const int cover_steps = static_cast<int>(infer_steps * cover_strength);
  const RepaintConfig repaint =
      ResolveRepaintConfig(request.repaint_mode, request.repaint_strength);
  const int injection_cutoff = static_cast<int>(
      std::nearbyint(repaint.injection_ratio * infer_steps));
  const bool guided = guidance > 1.0F;
  tm.guided = guided;
  ApgGuidance apg;
  std::vector<float> v, v_cond, v_uncond;
  // Live preview: the clean estimate of the current step is decoded (a short
  // window only) when the step is a candidate, someone wants it and the
  // previous preview is old enough; its time counts as diffusion time.
  const auto preview_interval =
      preview != nullptr
          ? std::chrono::duration<double>(preview->min_interval_seconds)
          : std::chrono::duration<double>(0);
  bool had_preview = false, switched = false;
  auto last_preview = std::chrono::steady_clock::now();
  for (int s = 0; s < infer_steps; ++s) {
    if (!report(0.5 + 0.35 * double(s) / infer_steps, "diffusion"))
      return Fail(error, "cancelled");
    if (cover_strength < 1.0F && s >= cover_steps && !switched) {
      switched = true;
      const int plain_rows =
          static_cast<int>(conditions_plain.size() / kConditionWidth);
      if (!m.dit->SetConditions(conditions_plain, plain_rows, context_plain,
                                frames, error))
        return false;
    }
    const float t = times[std::size_t(s)];
    const float t_next = times[std::size_t(s) + 1];
    if (!m.dit->Velocity(x, t, &v_cond, nullptr, error))
      return false;
    const bool apply_guidance = guided && t >= request.cfg_interval_start &&
                                t <= request.cfg_interval_end;
    if (apply_guidance) {
      if (!m.dit->Velocity(x, t, &v_uncond, nullptr, error, true))
        return false;
      apg.Apply(v_cond, v_uncond, guidance, frames, &v);
    } else {
      v = v_cond;
    }
    const int step = s + 1;
    if (preview != nullptr && preview->wanted &&
        MusicPreviewStep(step, infer_steps, preview->every) &&
        (!had_preview ||
         std::chrono::steady_clock::now() - last_preview >= preview_interval) &&
        preview->wanted()) {
      PublishPreview(*m.vae, *preview, x, v, t, frames, step, infer_steps);
      had_preview = true;
      last_preview = std::chrono::steady_clock::now();
    }
    const float dt = t - t_next;
    for (std::size_t i = 0; i < x.size(); ++i)
      x[i] -= v[i] * dt;
    if (!repaint_mask.empty() && s < injection_cutoff)
      RepaintInjection(x, clean_latents, repaint_mask, t_next, noise);
  }
  if (!repaint_mask.empty() && repaint.crossfade_frames > 0)
    RepaintBoundaryBlend(x, clean_latents, repaint_mask,
                         repaint.crossfade_frames);
  tm.diffusion_seconds = Since(start);
  if (!report(0.85, "vae"))
    return Fail(error, "cancelled");
  start = std::chrono::steady_clock::now();
  if (!m.vae->Decode(x, frames, audio, error))
    return false;
  // Repaint: the untouched parts come from the original waveform.
  if (TaskRepaints(task) && Lower(request.repaint_mode) != "aggressive")
    RepaintWaveformSplice(audio, source_audio, repaint_start, repaint_end,
                          kSampleRate, repaint.wav_crossfade_seconds);
  float peak = 0;
  for (float value : *audio)
    peak = std::max(peak, std::fabs(value));
  if (peak > 1.0F)
    for (float& value : *audio)
      value /= peak;
  tm.vae_seconds = Since(start);
  report(1.0, "done");
  return true;
}

}  // namespace gufo::models::ace_step
