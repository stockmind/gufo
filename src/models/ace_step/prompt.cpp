// ACE-Step 1.5 host side: model file layout, the text / lyric / LM prompts,
// the seeded noise and the WAV writer. CPU only (split from pipeline.cpp so
// `gufo serve music` request handling and the token oracle test build
// without HIP).
// Written with AI assistance (Claude Code).
#include "src/models/ace_step/pipeline.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <string_view>
#include <system_error>

#include <gufo/ext/json.hpp>

namespace gufo::models::ace_step {
namespace {

std::string OrNa(const std::string& value) {
  return value.empty() ? "N/A" : value;
}

bool MissingFile(std::string* error, const std::filesystem::path& path) {
  if (error != nullptr)
    *error = "ACE-Step model file missing: " + path.string();
  return false;
}

bool Fail(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
  return false;
}

std::string Lower(std::string_view text) {
  std::string out(text);
  for (char& c : out)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

bool Exists(const std::filesystem::path& path) {
  std::error_code ec;
  return std::filesystem::is_regular_file(path, ec);
}

// A `__metadata__` string of a safetensors file (8-byte header length, JSON
// header), read on the host so the variant is known without the HIP loader.
std::string SafetensorsMetadata(const std::filesystem::path& path,
                                const std::string& key) {
  std::ifstream in(path, std::ios::binary);
  std::uint8_t length_bytes[8];
  if (!in.read(reinterpret_cast<char*>(length_bytes), 8))
    return "";
  std::uint64_t length = 0;
  for (int i = 7; i >= 0; --i)
    length = (length << 8U) | length_bytes[i];
  if (length == 0 || length > (64ULL << 20U))
    return "";
  std::string header(static_cast<std::size_t>(length), '\0');
  if (!in.read(header.data(), static_cast<std::streamsize>(length)))
    return "";
  try {
    const json::Value doc = json::parse(header);
    const json::Value* metadata = doc.find("__metadata__");
    if (metadata == nullptr || !metadata->is_object())
      return "";
    return metadata->member_str(key);
  } catch (...) {
    return "";
  }
}

// The variant written by tools/ace_step/convert_hf_to_gufo.py
// (`gufo.variant`), the Hugging Face `model_version`, else the file name.
DitVariant DetectVariant(const std::filesystem::path& dit) {
  DitVariant variant = DitVariant::kTurbo;
  for (const char* key : {"gufo.variant", "model_version"}) {
    const std::string value = SafetensorsMetadata(dit, key);
    if (!value.empty() && GuessDitVariant(value, &variant))
      return variant;
  }
  if (GuessDitVariant(dit.filename().string(), &variant))
    return variant;
  return DitVariant::kTurbo;
}

}  // namespace

bool ModelPaths::FromRoot(const std::filesystem::path& root, ModelPaths* paths,
                          std::string* error) {
  return Resolve(root, "", "", "", paths, error);
}

bool ModelPaths::Resolve(const std::filesystem::path& root,
                         std::string_view dit, std::string_view lm,
                         std::string_view variant, ModelPaths* paths,
                         std::string* error) {
  paths->text = root / "text_encoder.safetensors";
  paths->vae = root / "vae.safetensors";
  paths->tokenizer = root / "tokenizer.json";
  paths->silence = root / "silence_latent.safetensors";
  // DiT: default, variant short name or a path.
  DitVariant named;
  if (dit.empty()) {
    paths->dit = root / "dit.safetensors";
  } else if (ParseDitVariant(dit, &named)) {
    paths->dit = root / ("dit-" + std::string(DitVariantName(named)) +
                         ".safetensors");
    if (named == DitVariant::kTurbo && !Exists(paths->dit))
      paths->dit = root / "dit.safetensors";
  } else {
    paths->dit = std::filesystem::path(dit);
    if (paths->dit.is_relative() && !Exists(paths->dit) &&
        Exists(root / paths->dit))
      paths->dit = root / paths->dit;
  }
  for (const auto* path : {&paths->dit, &paths->text, &paths->vae,
                           &paths->tokenizer, &paths->silence})
    if (!Exists(*path))
      return MissingFile(error, *path);
  // LM: default (when present), "none", a size or a path.
  const std::string lm_key = Lower(lm);
  paths->lm.clear();
  if (lm.empty()) {
    if (Exists(root / "lm.safetensors"))
      paths->lm = root / "lm.safetensors";
  } else if (lm_key == "none" || lm_key == "off" || lm_key == "no") {
    paths->lm.clear();
  } else if (lm_key == "0.6b" || lm_key == "1.7b" || lm_key == "4b") {
    paths->lm = root / ("lm-" + lm_key + ".safetensors");
    if (!Exists(paths->lm))
      return Fail(error, "ACE-Step LM file missing: " + paths->lm.string());
  } else {
    paths->lm = std::filesystem::path(lm);
    if (paths->lm.is_relative() && !Exists(paths->lm) &&
        Exists(root / paths->lm))
      paths->lm = root / paths->lm;
    if (!Exists(paths->lm))
      return Fail(error, "ACE-Step LM file missing: " + paths->lm.string());
  }
  if (variant.empty()) {
    paths->variant = DetectVariant(paths->dit);
  } else if (!ParseDitVariant(variant, &paths->variant)) {
    return Fail(error, "ACE-Step variant must be turbo, sft or base, not " +
                           std::string(variant));
  }
  return true;
}

std::string MusicTextPrompt(const MusicRequest& r, std::string_view instruction,
                            double duration) {
  const std::string metas =
      "- bpm: " + OrNa(r.bpm) + "\n- timesignature: " + OrNa(r.timesignature) +
      "\n- keyscale: " + OrNa(r.keyscale) +
      "\n- duration: " + std::to_string(static_cast<int>(duration)) +
      " seconds\n";
  return "# Instruction\n" + std::string(instruction) + "\n\n# Caption\n" +
         r.caption + "\n\n# Metas\n" + metas + "<|endoftext|>\n";
}

std::string MusicTextPrompt(const MusicRequest& r) {
  return MusicTextPrompt(r, DefaultMusicInstruction(), r.duration);
}

std::string MusicLyricPrompt(const MusicRequest& r) {
  const std::string lyrics = r.lyrics.empty() ? "[Instrumental]" : r.lyrics;
  return "# Languages\n" +
         (r.language.empty() ? std::string("en") : r.language) +
         "\n\n# Lyric\n" + lyrics + "<|endoftext|>";
}

namespace {

constexpr const char* kLmSystem =
    "<|im_start|>system\n# Instruction\nGenerate audio semantic tokens based "
    "on the given conditions:\n\n<|im_end|>\n<|im_start|>user\n";

std::string Instrumental(const std::string& lyrics) {
  return lyrics.empty() ? "[Instrumental]" : lyrics;
}

}  // namespace

std::string MusicLmPrompt(const MusicRequest& r) {
  // Sorted YAML of the user metadata, as ComfyUI writes it.
  std::string metas;
  if (!r.bpm.empty())
    metas += "bpm: " + r.bpm + "\n";
  metas +=
      "duration: " + std::to_string(static_cast<int>(std::ceil(r.duration))) +
      "\n";
  if (!r.keyscale.empty())
    metas += "keyscale: " + r.keyscale + "\n";
  if (!r.timesignature.empty())
    metas += "timesignature: " + r.timesignature + "\n";
  metas.pop_back();
  return std::string(kLmSystem) + "# Caption\n" + r.caption + "\n\n# Lyric\n" +
         Instrumental(r.lyrics) +
         "\n<|im_end|>\n<|im_start|>assistant\n<think>\n" + metas +
         "\n</think>\n\n";
}

std::string MusicLmUnconditionalPrompt() {
  return std::string(kLmSystem) +
         "NO USER INPUT<|im_end|>\n<|im_start|>assistant\n<think>\n\n"
         "</think>\n\n";
}

std::vector<float> MusicNoise(std::uint64_t seed, std::size_t count) {
  std::mt19937_64 engine(seed);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> noise(count);
  for (float& value : noise)
    value = normal(engine);
  return noise;
}

}  // namespace gufo::models::ace_step
