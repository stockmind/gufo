// gufo-ace-check: compares the ACE-Step port stage by stage with the torch
// oracle (ft/ace-step/tools/ace_oracle.py output: raw FP32 / INT32 files and
// manifest.json).
//
//   gufo-ace-check --oracle DIR --case N --text qwen_0.6b_ace15.safetensors
//       --dit acestep_v1.5_xl_turbo_bf16.safetensors --tokenizer tokenizer.json
//       [--stages tokens,text,cond,v0,steps]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include <gufo/ext/json.hpp>
#include "src/models/ace_step/model.hpp"
#include "src/models/ace_step/tokenizer.hpp"
#include "src/models/ace_step/vae.hpp"

namespace ace = gufo::models::ace_step;
namespace json = gufo::json;

namespace {

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), {});
}

template<typename T>
std::vector<T> ReadRaw(const std::filesystem::path& path) {
  const std::string bytes = ReadText(path);
  std::vector<T> out(bytes.size() / sizeof(T));
  std::memcpy(out.data(), bytes.data(), out.size() * sizeof(T));
  return out;
}

double Cosine(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size() || a.empty())
    return -2;
  double ab = 0, aa = 0, bb = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    ab += double(a[i]) * b[i];
    aa += double(a[i]) * a[i];
    bb += double(b[i]) * b[i];
  }
  return ab / (std::sqrt(aa * bb) + 1e-30);
}

double MaxAbs(const std::vector<float>& a, const std::vector<float>& b) {
  double m = 0;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i)
    m = std::max(m, std::fabs(double(a[i]) - b[i]));
  return m;
}

int failures = 0;

void Report(const char* name, const std::vector<float>& got,
            const std::vector<float>& want, double threshold = 0.999) {
  const double c = Cosine(got, want);
  const bool ok = c >= threshold;
  if (!ok)
    ++failures;
  std::printf("%-22s cos %.6f  max|d| %.4g  n %zu/%zu  %s\n", name, c,
              MaxAbs(got, want), got.size(), want.size(), ok ? "ok" : "FAIL");
  std::fflush(stdout);
}

double Seconds(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - since)
      .count();
}

}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i + 1 < argc; i += 2)
    args[argv[i]] = argv[i + 1];
  for (const char* required : {"--oracle", "--text", "--dit", "--tokenizer"})
    if (!args.count(required)) {
      std::fprintf(stderr,
                   "usage: gufo-ace-check --oracle DIR --case N --text F --dit "
                   "F --tokenizer F [--stages tokens,text,cond,v0,steps]\n");
      return 2;
    }
  const std::filesystem::path oracle = args["--oracle"];
  const std::string c = "c" + (args.count("--case") ? args["--case"] : "0");
  const std::string stages =
      args.count("--stages") ? args["--stages"] : "tokens,text,cond,v0,steps";
  auto want = [&](const char* stage) {
    return stages.find(stage) != std::string::npos;
  };
  auto f32 = [&](const std::string& name) {
    return ReadRaw<float>(oracle / (c + "_" + name + ".f32"));
  };
  auto ids = [&](const std::string& name) {
    const auto raw = ReadRaw<std::int32_t>(oracle / (c + "_" + name + ".i32"));
    return std::vector<std::uint32_t>(raw.begin(), raw.end());
  };
  const json::Value manifest = json::parse(ReadText(oracle / "manifest.json"));
  const int case_index = std::stoi(c.substr(1));
  const json::Value& info =
      manifest.find("report")->find("cases")->items()[case_index];
  std::string error;

  // Stage 1a: tokenizer.
  const auto text_ids = ids("text_ids");
  const auto lyric_ids = ids("lyric_ids");
  if (want("tokens")) {
    ace::Tokenizer tokenizer(args["--tokenizer"]);
    auto got_text = tokenizer.Encode(info.member_str("text_prompt"));
    got_text.push_back(151643);  // appended by the Qwen3-Embedding tokenizer
    auto got_lyric = tokenizer.Encode(info.member_str("lyric_prompt"));
    got_lyric.push_back(151643);
    const bool ok = got_text == text_ids && got_lyric == lyric_ids;
    if (!ok)
      ++failures;
    std::printf("%-22s text %zu/%zu lyric %zu/%zu  %s\n", "tokens",
                got_text.size(), text_ids.size(), got_lyric.size(),
                lyric_ids.size(), ok ? "identical" : "FAIL");
  }

  // Stage 1b: Qwen3-0.6B text states and lyric embeddings.
  if (want("text")) {
    auto start = std::chrono::steady_clock::now();
    auto encoder = ace::TextEncoder::Load(args["--text"], &error);
    if (!encoder) {
      std::fprintf(stderr, "text encoder: %s\n", error.c_str());
      return 1;
    }
    std::printf("text encoder loaded in %.1f s (%.2f GB)\n", Seconds(start),
                encoder->weight_bytes() / 1e9);
    std::vector<float> states, embeds;
    start = std::chrono::steady_clock::now();
    if (!encoder->Encode(text_ids, &states, &error) ||
        !encoder->Embed(lyric_ids, &embeds, &error)) {
      std::fprintf(stderr, "text encoder: %s\n", error.c_str());
      return 1;
    }
    std::printf("text encode %.3f s\n", Seconds(start));
    Report("text_states", states, f32("text_states"));
    Report("lyric_embeds", embeds, f32("lyric_embeds"));
  }

  // Stage 3: VAE decoder on the oracle's final latents.
  if (want("vae")) {
    if (!args.count("--vae")) {
      std::fprintf(stderr, "--vae required for the vae stage\n");
      return 2;
    }
    auto start = std::chrono::steady_clock::now();
    auto vae = ace::VaeDecoder::Load(args["--vae"], &error);
    if (!vae) {
      std::fprintf(stderr, "VAE: %s\n", error.c_str());
      return 1;
    }
    std::printf("VAE loaded in %.1f s\n", Seconds(start));
    const auto latents = f32("final");
    std::vector<float> audio;
    start = std::chrono::steady_clock::now();
    if (!vae->Decode(latents, int(latents.size() / 64), &audio, &error)) {
      std::fprintf(stderr, "VAE: %s\n", error.c_str());
      return 1;
    }
    std::printf("VAE decode %.2f s for %.1f s audio\n", Seconds(start),
                latents.size() / 64 / 25.0);
    Report("wav", audio, f32("wav"));
  }

  // Stage 5: the 5 Hz audio-code LM (greedy codes and first-step logits).
  if (want("lm")) {
    if (!args.count("--lm")) {
      std::fprintf(stderr, "--lm required for the lm stage\n");
      return 2;
    }
    auto start = std::chrono::steady_clock::now();
    auto lm = ace::CodeLm::Load(args["--lm"], &error);
    if (!lm) {
      std::fprintf(stderr, "LM: %s\n", error.c_str());
      return 1;
    }
    std::printf("LM loaded in %.1f s (%.2f GB)\n", Seconds(start),
                lm->weight_bytes() / 1e9);
    auto raw_ids = [&](const char* name) {
      const auto raw = ReadRaw<std::int32_t>(oracle / name);
      return std::vector<std::uint32_t>(raw.begin(), raw.end());
    };
    const auto cond = raw_ids("lm_ids.i32");
    const auto uncond = raw_ids("lm_uncond_ids.i32");
    if (args.count("--tokenizer")) {
      ace::Tokenizer tokenizer(args["--tokenizer"]);
      const auto prompt =
          manifest.find("report")->find("lm")->member_str("lm_prompt");
      const auto got = tokenizer.Encode(prompt);
      const bool same = got == cond;
      if (!same) {
        ++failures;
        std::size_t first = 0;
        while (first < got.size() && first < cond.size() &&
               got[first] == cond[first])
          ++first;
        std::printf("lm tokens differ at %zu (gufo %zu, oracle %zu):", first,
                    got.size(), cond.size());
        for (std::size_t i = first; i < first + 6; ++i)
          std::printf(" %u/%u", i < got.size() ? got[i] : 0U,
                      i < cond.size() ? cond[i] : 0U);
        std::printf("\n");
      }
      std::printf("%-22s %s\n", "lm_tokens", same ? "identical" : "FAIL");
    }
    ace::CodeSampling greedy;
    greedy.cfg = 1.0F;
    greedy.temperature = 0.0F;
    std::vector<int> codes;
    start = std::chrono::steady_clock::now();
    if (!lm->Generate(cond, uncond, 8, greedy, &codes, nullptr, &error)) {
      std::fprintf(stderr, "LM: %s\n", error.c_str());
      return 1;
    }
    Report("lm_logits0", lm->first_logits(),
           ReadRaw<float>(oracle / "lm_logits0.f32"));
    const auto want_codes = ReadRaw<std::int32_t>(oracle / "lm_greedy.i32");
    int same = 0;
    for (std::size_t i = 0; i < codes.size() && i < want_codes.size(); ++i)
      same += codes[i] == want_codes[i];
    std::printf("%-22s %d/%zu identical (%.1f codes/s)\n", "lm_greedy", same,
                want_codes.size(), lm->last_codes_per_second());
    ace::CodeSampling guided;
    guided.seed = 1;
    if (!lm->Generate(cond, uncond, 50, guided, &codes, nullptr, &error)) {
      std::fprintf(stderr, "LM: %s\n", error.c_str());
      return 1;
    }
    const auto& raw = lm->first_logits();
    std::vector<float> cfg(ace::CodeLm::kCodes);
    for (int k = 0; k < ace::CodeLm::kCodes; ++k)
      cfg[k] = raw[ace::CodeLm::kCodes + k] +
               2.0F * (raw[k] - raw[ace::CodeLm::kCodes + k]);
    const auto want_cfg = ReadRaw<float>(oracle / "lm_cfg_logits0.f32");
    const auto want_cond = ReadRaw<float>(oracle / "lm_logits0.f32");
    std::vector<float> want_uncond(want_cfg.size());
    for (std::size_t k = 0; k < want_cfg.size(); ++k)
      want_uncond[k] = 2.0F * want_cond[k] - want_cfg[k];
    Report("lm_cond_logits0",
           std::vector<float>(raw.begin(), raw.begin() + ace::CodeLm::kCodes),
           want_cond);
    Report("lm_uncond_logits0",
           std::vector<float>(raw.begin() + ace::CodeLm::kCodes, raw.end()),
           want_uncond);
    Report("lm_cfg_logits0", cfg, want_cfg, 0.0);
    // CFG amplifies cond - uncond; compare what sampling sees: the
    // distribution softmax(cfg / 0.85) (KL, top-p mass overlap) and the size
    // of cond - uncond against the BF16 error.
    auto softmax = [](const std::vector<float>& x) {
      double top = -1e30;
      for (float v : x)
        top = std::max(top, double(v) / 0.85);
      std::vector<double> p(x.size());
      double total = 0;
      for (std::size_t i = 0; i < x.size(); ++i)
        total += p[i] = std::exp(double(x[i]) / 0.85 - top);
      for (double& v : p)
        v /= total;
      return p;
    };
    const auto p_want = softmax(want_cfg), p_got = softmax(cfg);
    double kl = 0, overlap = 0;
    for (std::size_t i = 0; i < p_want.size(); ++i) {
      if (p_want[i] > 0 && p_got[i] > 0)
        kl += p_want[i] * std::log(p_want[i] / p_got[i]);
      overlap += std::min(p_want[i], p_got[i]);
    }
    double diff = 0, err = 0;
    for (std::size_t i = 0; i < want_cfg.size(); ++i) {
      diff += std::pow(double(want_cond[i]) - want_uncond[i], 2);
      err += std::pow(double(raw[i]) - want_cond[i], 2);
    }
    std::printf(
        "lm cfg distribution: KL %.4f, overlap %.4f, |cond-uncond| "
        "%.2f vs |cond error| %.2f\n",
        kl, overlap, std::sqrt(diff), std::sqrt(err));
    for (const std::vector<float>* values :
         {static_cast<const std::vector<float>*>(&cfg), &want_cfg}) {
      std::vector<int> order(values->size());
      for (std::size_t i = 0; i < order.size(); ++i)
        order[i] = static_cast<int>(i);
      std::partial_sort(
          order.begin(), order.begin() + 5, order.end(),
          [&](int a, int b) { return (*values)[a] > (*values)[b]; });
      std::printf("lm cfg top5 %s:", values == &cfg ? "gufo  " : "oracle");
      for (int i = 0; i < 5; ++i)
        std::printf(" %d(%.1f)", order[i], (*values)[order[i]]);
      std::printf("\n");
    }
    std::printf("lm cfg sampling: %.1f codes/s (2 sequences)\n",
                lm->last_codes_per_second());
  }

  if (!want("cond") && !want("v0") && !want("steps") && !want("hints"))
    return failures == 0 ? 0 : 1;
  auto start = std::chrono::steady_clock::now();
  auto dit = ace::Dit::Load(args["--dit"], &error);
  if (!dit) {
    std::fprintf(stderr, "DiT: %s\n", error.c_str());
    return 1;
  }
  std::printf("DiT loaded in %.1f s (%.2f GB)\n", Seconds(start),
              dit->weight_bytes() / 1e9);

  // Stage 5b: audio-code detokenizer.
  if (want("hints")) {
    const auto raw = ReadRaw<std::int32_t>(oracle / "hint_codes.i32");
    const std::vector<int> codes(raw.begin(), raw.end());
    const auto expected = ReadRaw<float>(oracle / "hint_latents.f32");
    std::vector<float> hints;
    if (!dit->CodeHints(codes, int(expected.size() / 64), &hints, &error)) {
      std::fprintf(stderr, "hints: %s\n", error.c_str());
      return 1;
    }
    Report("hint_latents", hints, expected);
  }

  // Stage 2a: condition encoder on the oracle text states.
  const auto oracle_enc = f32("enc");
  if (want("cond")) {
    const auto text = f32("text_states");
    const auto lyric = f32("lyric_embeds");
    // The oracle's silence latent: first 750 frames are the timbre input.
    auto silence = ReadRaw<float>(oracle / "silence_timbre.f32");
    std::vector<float> enc;
    start = std::chrono::steady_clock::now();
    if (!dit->EncodeConditions(text, int(text.size() / 1024), lyric,
                               int(lyric.size() / 1024), silence,
                               int(silence.size() / 64), &enc, &error)) {
      std::fprintf(stderr, "conditions: %s\n", error.c_str());
      return 1;
    }
    std::printf("condition encoder %.3f s\n", Seconds(start));
    Report("enc", enc, oracle_enc);
    const int lyric_rows = int(lyric.size() / 1024);
    auto slice = [&](int first, int rows) {
      return std::vector<float>(enc.begin() + std::size_t(first) * 2048,
                                enc.begin() + std::size_t(first + rows) * 2048);
    };
    Report("enc_lyric", slice(0, lyric_rows), f32("enc_lyric"));
    Report("enc_timbre", slice(lyric_rows, 1), f32("enc_timbre"));
    Report("enc_text", slice(lyric_rows + 1, int(text.size() / 1024)),
           f32("enc_text"));
  }

  // Stage 2b: one decoder call and the 8-step sampler on oracle inputs.
  const auto context = f32("context");
  const auto noise = f32("noise");
  const int frames = int(noise.size() / 64);
  if (!dit->SetConditions(oracle_enc, int(oracle_enc.size() / 2048), context,
                          frames, &error)) {
    std::fprintf(stderr, "set conditions: %s\n", error.c_str());
    return 1;
  }
  if (want("v0")) {
    std::vector<float> v;
    ace::DitProbes probes;
    if (!dit->Velocity(noise, 1.0F, &v, &probes, &error)) {
      std::fprintf(stderr, "velocity: %s\n", error.c_str());
      return 1;
    }
    Report("v0_patch", probes.patch, f32("v0_patch"));
    Report("v0_layer0", probes.layer0, f32("v0_layer0"));
    Report("v0_layer31", probes.layer31, f32("v0_layer31"));
    Report("v0", v, f32("v0"));
    // Timing without probes.
    if (!dit->Velocity(noise, 1.0F, &v, nullptr, &error))
      return 1;
    std::printf("velocity %d frames: %.1f ms\n", frames,
                dit->last_velocity_ms());
  }
  if (want("steps")) {
    const auto times = ace::TurboTimesteps(8, 3.0F);
    std::vector<float> x = noise, v;
    start = std::chrono::steady_clock::now();
    for (std::size_t s = 0; s < times.size(); ++s) {
      if (!dit->Velocity(x, times[s], &v, nullptr, &error)) {
        std::fprintf(stderr, "velocity: %s\n", error.c_str());
        return 1;
      }
      const float dt =
          s + 1 < times.size() ? times[s] - times[s + 1] : times[s];
      for (std::size_t i = 0; i < x.size(); ++i)
        x[i] -= v[i] * dt;
      const std::string name = "step" + std::to_string(s) + "_x";
      Report(name.c_str(), x, f32(name));
    }
    std::printf("8 steps %.2f s\n", Seconds(start));
    Report("final", x, f32("final"));
    std::ofstream(oracle / (c + "_gufo_final.f32"), std::ios::binary)
        .write(reinterpret_cast<const char*>(x.data()),
               std::streamsize(x.size() * 4));
  }
  std::printf("%s\n", failures == 0 ? "ALL OK" : "FAILURES");
  return failures == 0 ? 0 : 1;
}
