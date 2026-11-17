#ifndef GUFO_MODELS_ACE_STEP_TOKENIZER_HPP_
#define GUFO_MODELS_ACE_STEP_TOKENIZER_HPP_

#include <filesystem>
#include <memory>
#include <string_view>
#include <vector>

#include "src/families/qwen/text/tokenizer.hpp"

namespace gufo::models::ace_step {

class Tokenizer {
public:
  /// Reads a Qwen3 `tokenizer.json` (byte-level BPE, NFC, Split + ByteLevel).
  explicit Tokenizer(const std::filesystem::path& tokenizer_json);
  std::vector<std::uint32_t> Encode(std::string_view input) const;

private:
  std::unique_ptr<tokenization::QwenTokenizer> bpe_;
  std::unordered_map<std::string, std::uint32_t> specials_;
  std::string pattern_;
};

}  // namespace gufo::models::ace_step
#endif
