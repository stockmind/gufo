// gufo-ace: music with ACE-Step 1.5 XL (turbo / sft / base).
//
//   gufo-ace --model ROOT --caption "..." [--lyrics "..." | --lyrics-file F]
//       [--duration 30] [--seed 42] [--bpm 120] [--key "A minor"]
//       [--timesignature 4] [--language en] [--steps N] [--shift S]
//       [--guidance G] [--dit FILE|turbo|sft|base] [--lm FILE|none|0.6b|..]
//       [--variant turbo|sft|base] [--task text2music|repaint|cover|
//       cover-nofsq|extract|lego|complete] [--source F.wav]
//       [--reference F.wav] [--repaint-start S] [--repaint-end S]
//       [--track NAME] [--repeat N] --out song.wav
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "src/models/ace_step/audio_io.hpp"
#include "src/models/ace_step/pipeline.hpp"
#include "src/models/ace_step/vae.hpp"

namespace ace = gufo::models::ace_step;

namespace {

bool ReadWav(const std::string& path, std::vector<float>* planar) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::fprintf(stderr, "cannot read %s\n", path.c_str());
    return false;
  }
  const std::string bytes((std::istreambuf_iterator<char>(in)), {});
  ace::StereoAudio audio;
  std::string error;
  if (!ace::DecodeWavStereo(
          std::span<const std::uint8_t>(
              reinterpret_cast<const std::uint8_t*>(bytes.data()),
              bytes.size()),
          &audio, &error)) {
    std::fprintf(stderr, "%s: %s\n", path.c_str(), error.c_str());
    return false;
  }
  ace::NormalizeStereo48k(&audio);
  *planar = std::move(audio.planar);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i + 1 < argc; i += 2)
    args[argv[i]] = argv[i + 1];
  if (!args.count("--model") || !args.count("--caption") ||
      !args.count("--out")) {
    std::fprintf(stderr,
                 "usage: gufo-ace --model ROOT --caption TEXT [--lyrics TEXT | "
                 "--lyrics-file F] [--duration S] [--seed N] [--bpm N] [--key "
                 "K] [--timesignature N] [--language L] [--steps N] [--shift "
                 "S] [--guidance G] [--dit FILE|turbo|sft|base] [--lm "
                 "FILE|none|0.6b|1.7b|4b] [--variant V] [--task T] [--source "
                 "F.wav] [--reference F.wav] [--repaint-start S] "
                 "[--repaint-end S] [--track NAME] [--repeat N] --out F.wav\n");
    return 2;
  }
  std::string error;
  ace::ModelPaths paths;
  if (!ace::ModelPaths::Resolve(args["--model"], args["--dit"], args["--lm"],
                                args["--variant"], &paths, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  auto start = std::chrono::steady_clock::now();
  auto pipeline = ace::MusicPipeline::Load(paths, &error);
  if (!pipeline) {
    std::fprintf(stderr, "load: %s\n", error.c_str());
    return 1;
  }
  std::printf(
      "loaded %s in %.1f s\n",
      std::string(ace::DitVariantName(pipeline->variant())).c_str(),
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count());
  ace::MusicRequest request;
  request.caption = args["--caption"];
  if (args.count("--lyrics-file")) {
    std::ifstream in(args["--lyrics-file"], std::ios::binary);
    request.lyrics.assign(std::istreambuf_iterator<char>(in), {});
  } else if (args.count("--lyrics")) {
    request.lyrics = args["--lyrics"];
  }
  if (args.count("--duration"))
    request.duration = std::stod(args["--duration"]);
  if (args.count("--seed"))
    request.seed = std::stoull(args["--seed"]);
  if (args.count("--steps"))
    request.steps = std::stoi(args["--steps"]);
  if (args.count("--shift"))
    request.shift = std::stof(args["--shift"]);
  if (args.count("--guidance"))
    request.guidance_scale = std::stof(args["--guidance"]);
  if (args.count("--bpm"))
    request.bpm = args["--bpm"];
  if (args.count("--key"))
    request.keyscale = args["--key"];
  if (args.count("--timesignature"))
    request.timesignature = args["--timesignature"];
  if (args.count("--language"))
    request.language = args["--language"];
  if (args.count("--task") && !ace::ParseMusicTask(args["--task"], &request.task)) {
    std::fprintf(stderr, "unknown task %s\n", args["--task"].c_str());
    return 2;
  }
  if (args.count("--source") && !ReadWav(args["--source"], &request.source_audio))
    return 1;
  if (args.count("--reference") &&
      !ReadWav(args["--reference"], &request.reference_audio))
    return 1;
  if (args.count("--repaint-start"))
    request.repaint_start = std::stod(args["--repaint-start"]);
  if (args.count("--repaint-end"))
    request.repaint_end = std::stod(args["--repaint-end"]);
  if (args.count("--track"))
    request.track = args["--track"];
  const int repeat = args.count("--repeat") ? std::stoi(args["--repeat"]) : 1;
  std::vector<float> audio;
  for (int r = 0; r < repeat; ++r) {
    ace::MusicTelemetry t;
    start = std::chrono::steady_clock::now();
    if (!pipeline->Generate(request, &audio, &t, nullptr, &error)) {
      std::fprintf(stderr, "generate: %s\n", error.c_str());
      return 1;
    }
    const double total =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    const double seconds = double(t.frames) / ace::kLatentFramesPerSecond;
    std::printf(
        "%s %.1f s audio (%d frames, %d steps%s, %d text / %d lyric tokens): "
        "text %.2f s, encode %.2f s, conditions %.2f s, diffusion %.2f s, vae "
        "%.2f s, total %.2f s (%.1fx real time)\n",
        std::string(ace::MusicTaskName(request.task)).c_str(), seconds,
        t.frames, t.steps, t.guided ? " guided" : "", t.text_tokens,
        t.lyric_tokens, t.text_seconds, t.encode_seconds, t.condition_seconds,
        t.diffusion_seconds, t.vae_seconds, total, seconds / total);
  }
  const std::string wav = ace::EncodeStereoWav(audio, ace::kSampleRate);
  std::ofstream(args["--out"], std::ios::binary)
      .write(wav.data(), std::streamsize(wav.size()));
  std::printf("wrote %s\n", args["--out"].c_str());
  return 0;
}
