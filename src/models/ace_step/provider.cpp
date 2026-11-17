// ACE-Step 1.5 XL music provider: maps the gufo music capability onto the
// ace_step::MusicPipeline ported from gufo-studio-windows.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <gufo/ext/generated/ace_step_constants.hpp>
#include <gufo/ext/json.hpp>
#include <gufo/ext/music.hpp>
#include <gufo/ext/options.hpp>
#include <gufo/ext/registry.hpp>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/ace_step/audio_io.hpp"
#include "src/models/ace_step/pipeline.hpp"

namespace gufo::models::ace_step {

// Registered by extension.cpp.
bool ProbeAceStep(std::string_view model_root);
bool ResolveAceStepArtifacts(std::string_view model_root, std::string* error);

namespace {

// Optional source/reference audio for the edit tasks. The HTTP layer passes a
// path inside the container (the queue writes uploads into the mounted job
// root), which keeps large audio out of the JSON body.
bool LoadPlanarWav(const std::string& path, std::vector<float>* planar,
                   std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot read audio: " + path;
    return false;
  }
  const std::string bytes((std::istreambuf_iterator<char>(in)), {});
  StereoAudio audio;
  std::string decode_error;
  if (!DecodeWavStereo(std::span<const std::uint8_t>(
                           reinterpret_cast<const std::uint8_t*>(bytes.data()),
                           bytes.size()),
                       &audio, &decode_error)) {
    *error = path + ": " + decode_error;
    return false;
  }
  NormalizeStereo48k(&audio);
  *planar = std::move(audio.planar);
  return true;
}

MusicTask ParseTask(std::string_view name) {
  if (name == "repaint")
    return MusicTask::kRepaint;
  if (name == "cover")
    return MusicTask::kCover;
  if (name == "cover-nofsq")
    return MusicTask::kCoverNoFsq;
  if (name == "extract")
    return MusicTask::kExtract;
  if (name == "lego")
    return MusicTask::kLego;
  if (name == "complete")
    return MusicTask::kComplete;
  return MusicTask::kText2Music;
}

constexpr std::uint32_t kSampleRate = 48000;
constexpr double kMaxSeconds = 600.0;
constexpr double kDefaultSeconds = 30.0;

std::string Option(const ext::ParsedOptions& options, const char* name) {
  const std::string* value = ext::FindOption<std::string>(options, name);
  return value == nullptr ? std::string() : *value;
}

double Milliseconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

std::string Telemetry(const MusicTelemetry& telemetry, double total_ms) {
  auto number = [](double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.3f", value);
    return std::string(buffer);
  };
  std::string json = "{";
  json += "\"text_ms\":" + number(telemetry.text_seconds * 1000.0);
  json += ",\"encode_ms\":" + number(telemetry.encode_seconds * 1000.0);
  json += ",\"condition_ms\":" + number(telemetry.condition_seconds * 1000.0);
  json += ",\"diffusion_ms\":" + number(telemetry.diffusion_seconds * 1000.0);
  json += ",\"vae_ms\":" + number(telemetry.vae_seconds * 1000.0);
  json += ",\"lm_ms\":" + number(telemetry.lm_seconds * 1000.0);
  json += ",\"codes\":" + std::to_string(telemetry.codes);
  json += ",\"frames\":" + std::to_string(telemetry.frames);
  json += ",\"steps\":" + std::to_string(telemetry.steps);
  json += ",\"text_tokens\":" + std::to_string(telemetry.text_tokens);
  json += ",\"lyric_tokens\":" + std::to_string(telemetry.lyric_tokens);
  json += ",\"guided\":";
  json += telemetry.guided ? "true" : "false";
  json += ",\"total_ms\":" + number(total_ms);
  json += "}";
  return json;
}

class AceStepRunner final : public ext::MusicRunner {
public:
  AceStepRunner(std::unique_ptr<MusicPipeline> pipeline, DitVariant variant)
      : pipeline_(std::move(pipeline)), variant_(variant) {}

  ext::MusicCapabilities Capabilities() const override {
    ext::MusicCapabilities capabilities;
    capabilities.lyrics = ext::MusicSupport::kOptional;
    capabilities.tempo_key_meter = ext::MusicSupport::kOptional;
    capabilities.language = ext::MusicSupport::kOptional;
    capabilities.guidance = true;
    capabilities.steps = true;
    capabilities.duration = ext::MusicDurationControl::kMaximum;
    capabilities.max_seconds = kMaxSeconds;
    return capabilities;
  }

  std::uint32_t SampleRate() const override { return kSampleRate; }

  std::string ModelId() const override {
    switch (variant_) {
      case DitVariant::kSft:
        return "ace-step-1.5-xl-sft";
      case DitVariant::kBase:
        return "ace-step-1.5-xl-base";
      case DitVariant::kTurbo:
      default:
        return "ace-step-1.5-xl-turbo";
    }
  }

  bool Generate(const ext::MusicRequest& request,
                const ext::CancellationCheck& is_cancelled,
                const ext::MusicProgress& progress, ext::MusicResult* result,
                std::string* error) override {
    const auto start = std::chrono::steady_clock::now();
    MusicRequest mapped;
    mapped.caption = request.prompt;
    mapped.lyrics = request.lyrics;
    if (request.language.has_value())
      mapped.language = *request.language;
    if (request.bpm.has_value()) {
      char buffer[32];
      std::snprintf(buffer, sizeof(buffer), "%.3f", *request.bpm);
      mapped.bpm = buffer;
    }
    if (request.key.has_value())
      mapped.keyscale = *request.key;
    if (request.beats_per_bar.has_value())
      mapped.timesignature = std::to_string(*request.beats_per_bar);
    mapped.duration = std::clamp(
        request.duration_seconds.value_or(kDefaultSeconds), 1.0, kMaxSeconds);
    mapped.seed = request.seed;
    if (request.steps.has_value())
      mapped.steps = *request.steps;
    if (request.guidance.has_value())
      mapped.guidance_scale = *request.guidance;
    mapped.task = MusicTask::kText2Music;

    // Advanced controls travel in provider_json (gufo.options): the edit tasks
    // and the sampler knobs the CLI exposes. Anything unknown is ignored.
    if (!request.provider_json.empty()) {
      json::Value options;
      try {
        options = json::parse(request.provider_json);
      } catch (const std::exception& exception) {
        if (error != nullptr)
          *error = std::string("options: ") + exception.what();
        return false;
      }
      const auto* task = options.find("task");
      if (task != nullptr && task->is_string())
        mapped.task = ParseTask(task->str());
      if (const auto* value = options.find("shift");
          value != nullptr && value->is_number())
        mapped.shift = static_cast<float>(value->as_double(0.0));
      if (const auto* value = options.find("repaint_start");
          value != nullptr && value->is_number())
        mapped.repaint_start = value->as_double(0.0);
      if (const auto* value = options.find("repaint_end");
          value != nullptr && value->is_number())
        mapped.repaint_end = value->as_double(-1.0);
      if (const auto* value = options.find("repaint_mode");
          value != nullptr && value->is_string())
        mapped.repaint_mode = value->str();
      if (const auto* value = options.find("repaint_strength");
          value != nullptr && value->is_number())
        mapped.repaint_strength = static_cast<float>(value->as_double(0.5));
      if (const auto* value = options.find("cover_strength");
          value != nullptr && value->is_number())
        mapped.cover_strength = static_cast<float>(value->as_double(1.0));
      if (const auto* value = options.find("cover_noise_strength");
          value != nullptr && value->is_number())
        mapped.cover_noise_strength = static_cast<float>(value->as_double(0.0));
      if (const auto* value = options.find("track");
          value != nullptr && value->is_string())
        mapped.track = value->str();
      if (mapped.task != MusicTask::kText2Music) {
        const auto* source = options.find("source");
        if (source == nullptr || !source->is_string() ||
            source->str().empty()) {
          if (error != nullptr)
            *error = "this task needs a source audio path";
          return false;
        }
        if (!LoadPlanarWav(source->str(), &mapped.source_audio, error))
          return false;
        if (const auto* reference = options.find("reference");
            reference != nullptr && reference->is_string() &&
            !reference->str().empty()) {
          if (!LoadPlanarWav(reference->str(), &mapped.reference_audio, error))
            return false;
        }
      }
    }

    std::vector<float> planar;
    MusicTelemetry telemetry;
    const MusicProgress step = [&](double fraction, std::string_view phase) {
      if (is_cancelled && is_cancelled())
        return false;
      if (!progress)
        return true;
      const double clamped = std::clamp(fraction, 0.0, 1.0);
      return progress(phase, static_cast<std::uint64_t>(clamped * 1000.0),
                      1000);
    };
    try {
      if (!pipeline_->Generate(mapped, &planar, &telemetry, step, error))
        return false;
    } catch (const std::exception& exception) {
      if (error != nullptr)
        *error = exception.what();
      return false;
    }
    if (planar.size() < 2)
      planar.assign(2, 0.0F);
    const std::size_t frames = planar.size() / 2;
    result->audio.sample_rate = kSampleRate;
    result->audio.channels = 2;
    result->audio.samples.resize(frames * 2);
    for (std::size_t i = 0; i < frames; ++i) {
      result->audio.samples[2 * i] = std::clamp(planar[i], -1.0F, 1.0F);
      result->audio.samples[2 * i + 1] =
          std::clamp(planar[frames + i], -1.0F, 1.0F);
    }
    result->finish = ext::MusicFinish::kComplete;
    result->total_ms = Milliseconds(start);
    result->telemetry_json = Telemetry(telemetry, result->total_ms);
    return true;
  }

private:
  std::unique_ptr<MusicPipeline> pipeline_;
  DitVariant variant_;
};

std::unique_ptr<ext::MusicRunner> CreateAceStepRunner(
    const ext::MusicRunnerOptions& options, std::string* error) {
  ModelPaths paths;
  if (!ModelPaths::Resolve(options.model_root, Option(options.options, "--dit"),
                           Option(options.options, "--lm"),
                           Option(options.options, "--variant"), &paths, error))
    return nullptr;
  auto pipeline = MusicPipeline::Load(paths, error);
  if (!pipeline)
    return nullptr;
  return std::make_unique<AceStepRunner>(std::move(pipeline),
                                         pipeline->variant());
}

}  // namespace

}  // namespace gufo::models::ace_step

void ace_step_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_ace_step::kProvider_ace_step);
  using namespace gufo::models::ace_step;
  registry.AddMusicProvider({
      .id = "ace-step",
      .probe = &ProbeAceStep,
      .resolve = &ResolveAceStepArtifacts,
      .create_with_options = &CreateAceStepRunner,
  });
}
