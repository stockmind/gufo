// ACE-Step 1.5 XL turbo: Qwen3-0.6B text encoder, condition encoder and DiT
// decoder. Every block is a Qwen3-style transformer layer (QK-RMSNorm, RoPE
// theta 1e6 over rotate-half pairs, grouped heads of 128, SwiGLU); the DiT
// adds AdaLN modulation and cross-attention. BF16 weights and GEMM operands,
// FP32 residual stream, norms and accumulation.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <random>
#include <string>
#include <utility>

#include "src/models/ace_step/attention.hip.hpp"
#include "src/models/ace_step/gemm.hpp"
#include "src/models/ace_step/kernels.hpp"
#include "src/models/ace_step/model.hpp"
#include "src/models/ace_step/sampler.hpp"
#include "src/models/ace_step/weights.hpp"

namespace gufo::models::ace_step {
namespace {

constexpr int kHeadDim = 128;
constexpr int kWindow = 128;

struct Shape {
  int width{0};
  int heads{0};
  int kv_heads{0};
  int ffn{0};
  [[nodiscard]] int q_dim() const { return heads * kHeadDim; }
  [[nodiscard]] int kv_dim() const { return kv_heads * kHeadDim; }
};

constexpr Shape kTextShape{1024, 16, 8, 3072};
constexpr Shape kEncoderShape{2048, 16, 8, 6144};
constexpr Shape kDitShape{2560, 32, 8, 9728};
constexpr int kTextLayers = 28;
constexpr int kLyricLayers = 8;
constexpr int kTimbreLayers = 4;
constexpr int kDitLayers = 32;
constexpr int kFrequencies = 256;

struct Attention {
  DeviceBuffer qkv;  // self: q | k | v rows [q_dim + 2 kv_dim, width]
  DeviceBuffer q;    // cross: [q_dim, width]
  DeviceBuffer kv;   // cross: k | v rows [2 kv_dim, width]
  DeviceBuffer q_norm, k_norm, out;
};

struct Mlp {
  DeviceBuffer gate_up;  // gate | up rows [2 ffn, width]
  DeviceBuffer down;     // [width, ffn]
};

struct EncoderLayer {
  DeviceBuffer input_norm, post_norm;
  Attention attn;
  Mlp mlp;
};

struct DitLayer {
  DeviceBuffer self_norm, cross_norm, mlp_norm;
  DeviceBuffer table;  // BF16 [6, 2560]
  Attention self_attn, cross_attn;
  Mlp mlp;
  DeviceBuffer cross_key;     // head-major [8][rows][128]
  DeviceBuffer cross_packed;  // packed values [8][128][PaddedRows(rows)]
  DeviceBuffer null_key;      // the same for the null condition (1 row)
  DeviceBuffer null_packed;
};

// Stacks the rows of `parts` (each [rows_i, columns]) into one BF16 matrix.
bool LoadStacked(SafetensorsFile& file, DeviceBuffer& buffer,
                 std::initializer_list<std::pair<std::string, int>> parts,
                 int columns, std::string* error) {
  std::size_t rows = 0;
  for (const auto& part : parts)
    rows += part.second;
  if (!buffer.Allocate(rows * columns * 2, error))
    return false;
  std::size_t row = 0;
  for (const auto& [name, part_rows] : parts) {
    if (!file.Upload(name, static_cast<std::size_t>(part_rows) * columns,
                     buffer.bf16() + row * columns, true, error))
      return false;
    row += part_rows;
  }
  return true;
}

bool LoadAttention(SafetensorsFile& file, const std::string& p, const Shape& s,
                   bool cross, Attention& a, std::string* error) {
  const int w = s.width;
  if (cross) {
    if (!file.Load(p + "q_proj.weight", std::size_t(s.q_dim()) * w, a.q, true,
                   error) ||
        !LoadStacked(file, a.kv,
                     {{p + "k_proj.weight", s.kv_dim()},
                      {p + "v_proj.weight", s.kv_dim()}},
                     w, error))
      return false;
  } else if (!LoadStacked(file, a.qkv,
                          {{p + "q_proj.weight", s.q_dim()},
                           {p + "k_proj.weight", s.kv_dim()},
                           {p + "v_proj.weight", s.kv_dim()}},
                          w, error)) {
    return false;
  }
  return file.Load(p + "q_norm.weight", kHeadDim, a.q_norm, true, error) &&
         file.Load(p + "k_norm.weight", kHeadDim, a.k_norm, true, error) &&
         file.Load(p + "o_proj.weight", std::size_t(w) * s.q_dim(), a.out, true,
                   error);
}

bool LoadMlp(SafetensorsFile& file, const std::string& p, const Shape& s,
             Mlp& m, std::string* error) {
  return LoadStacked(
             file, m.gate_up,
             {{p + "gate_proj.weight", s.ffn}, {p + "up_proj.weight", s.ffn}},
             s.width, error) &&
         file.Load(p + "down_proj.weight", std::size_t(s.width) * s.ffn, m.down,
                   true, error);
}

bool LoadEncoderLayers(SafetensorsFile& file, const std::string& prefix,
                       int count, const Shape& s,
                       std::vector<EncoderLayer>& layers, std::string* error) {
  layers.resize(count);
  for (int i = 0; i < count; ++i) {
    const std::string p = prefix + "layers." + std::to_string(i) + ".";
    EncoderLayer& l = layers[i];
    if (!file.Load(p + "input_layernorm.weight", s.width, l.input_norm, true,
                   error) ||
        !file.Load(p + "post_attention_layernorm.weight", s.width, l.post_norm,
                   true, error) ||
        !LoadAttention(file, p + "self_attn.", s, false, l.attn, error) ||
        !LoadMlp(file, p + "mlp.", s, l.mlp, error))
      return false;
  }
  return true;
}

// Scratch buffers and the primitive sequence shared by every stack.
struct Engine {
  hipStream_t stream{nullptr};
  LinearGemm gemm;
  DeviceBuffer normed, qkv, query, key, value, packed, attn, proj, ffn, act;
  DeviceBuffer ids;

  ~Engine() {
    if (stream != nullptr) {
      (void)hipStreamSynchronize(stream);
      (void)hipStreamDestroy(stream);
    }
  }

  bool Init(std::string* error) {
    return Check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
                 "stream", error) &&
           gemm.Init(stream, error);
  }

  bool Reserve(int rows, const Shape& s, std::string* error) {
    const std::size_t r = rows, w = s.width, q = s.q_dim(), kv = s.kv_dim();
    const std::size_t f = s.ffn;
    return normed.Reserve(r * w * 2, error) &&
           qkv.Reserve(r * (q + 2 * kv) * 2, error) &&
           query.Reserve(r * q * 2, error) && key.Reserve(r * kv * 2, error) &&
           value.Reserve(r * kv * 2, error) &&
           packed.Reserve(std::size_t(PaddedRows(rows)) * kv * 2, error) &&
           attn.Reserve(r * q * 2, error) && proj.Reserve(r * w * 2, error) &&
           ffn.Reserve(r * 2 * f * 2, error) && act.Reserve(r * f * 2, error);
  }

  bool Gemm(const DeviceBuffer& weight, const DeviceBuffer* bias,
            const std::uint16_t* in, std::uint16_t* out, int tokens, int rows,
            int columns, std::string* error) {
    return gemm.Run(weight.bf16(), bias != nullptr ? bias->bf16() : nullptr, in,
                    out, tokens, rows, columns, error);
  }

  // proj = o_proj(attention(normed)) for self-attention. With mask.group,
  // rows form independent sequences of that length (RoPE restarts).
  bool SelfAttention(const Attention& a, const Shape& s, int rows,
                     AttentionMask mask, std::string* error) {
    const int stride = s.q_dim() + 2 * s.kv_dim();
    if (!Gemm(a.qkv, nullptr, normed.bf16(), qkv.bf16(), rows, stride, s.width,
              error))
      return false;
    const int period = static_cast<int>(mask.group);
    LaunchHeads(qkv.bf16(), stride, 0, rows, s.heads, a.q_norm.bf16(), true,
                query.bf16(), stream, nullptr, period);
    LaunchHeads(qkv.bf16(), stride, s.q_dim(), rows, s.kv_heads,
                a.k_norm.bf16(), true, key.bf16(), stream, nullptr, period);
    LaunchHeads(qkv.bf16(), stride, s.q_dim() + s.kv_dim(), rows, s.kv_heads,
                nullptr, false, value.bf16(), stream);
    LaunchPackValues(value.bf16(), packed.bf16(), rows, s.kv_heads, stream);
    LaunchAttention(query.bf16(), key.bf16(), packed.bf16(), attn.bf16(), rows,
                    rows, s.heads, s.kv_heads, mask, stream);
    return Gemm(a.out, nullptr, attn.bf16(), proj.bf16(), rows, s.width,
                s.q_dim(), error);
  }

  // proj = down(silu(gate(normed)) * up(normed)).
  bool FeedForward(const Mlp& m, const Shape& s, int rows, std::string* error) {
    if (!Gemm(m.gate_up, nullptr, normed.bf16(), ffn.bf16(), rows, 2 * s.ffn,
              s.width, error))
      return false;
    LaunchSwiGlu(ffn.bf16(), rows, s.ffn, act.bf16(), stream);
    return Gemm(m.down, nullptr, act.bf16(), proj.bf16(), rows, s.width, s.ffn,
                error);
  }

  // Pre-norm layers on the FP32 residual `hidden` [rows, width]; even layers
  // use the +-128 window when `windowed`.
  bool EncoderLayers(const std::vector<EncoderLayer>& layers, const Shape& s,
                     float* hidden, int rows, bool windowed, bool causal,
                     std::string* error, int group = 0) {
    if (!Reserve(rows, s, error))
      return false;
    for (std::size_t i = 0; i < layers.size(); ++i) {
      const EncoderLayer& l = layers[i];
      AttentionMask mask;
      mask.causal = causal;
      mask.group = static_cast<std::uint32_t>(group);
      mask.window = windowed && i % 2 == 0 ? kWindow : 0;
      LaunchRmsNorm(hidden, l.input_norm.bf16(), nullptr, nullptr, rows,
                    s.width, normed.bf16(), nullptr, stream);
      if (!SelfAttention(l.attn, s, rows, mask, error))
        return false;
      LaunchResidual(hidden, proj.bf16(), nullptr, rows, s.width, stream);
      LaunchRmsNorm(hidden, l.post_norm.bf16(), nullptr, nullptr, rows, s.width,
                    normed.bf16(), nullptr, stream);
      if (!FeedForward(l.mlp, s, rows, error))
        return false;
      LaunchResidual(hidden, proj.bf16(), nullptr, rows, s.width, stream);
    }
    return true;
  }

  // Synchronous: callers pass short-lived host vectors.
  bool Upload(DeviceBuffer& buffer, const void* data, std::size_t bytes,
              std::string* error) {
    return buffer.Reserve(bytes, error) &&
           Check(hipMemcpyAsync(buffer.get(), data, bytes,
                                hipMemcpyHostToDevice, stream),
                 "upload", error) &&
           Check(hipStreamSynchronize(stream), "upload", error);
  }

  bool Download(const void* device, std::size_t bytes, void* host,
                std::string* error) {
    return Check(hipMemcpyAsync(host, device, bytes, hipMemcpyDeviceToHost,
                                stream),
                 "download", error) &&
           Check(hipStreamSynchronize(stream), "download", error);
  }

  // Uploads FP32 rows and converts them to BF16 in `target`.
  bool UploadBf16(std::span<const float> data, DeviceBuffer& staging,
                  DeviceBuffer& target, std::string* error) {
    return Upload(staging, data.data(), data.size() * 4, error) &&
           target.Reserve(data.size() * 2, error) &&
           (LaunchToBf16(staging.f32(), data.size(), target.bf16(), stream),
            true);
  }
};

}  // namespace

// ----------------------------------------------------------------- text

struct TextEncoder::Impl {
  Engine engine;
  DeviceBuffer embed;  // [151669, 1024]
  std::vector<EncoderLayer> layers;
  DeviceBuffer norm;
  DeviceBuffer hidden;
  std::size_t weight_bytes{0};
  std::size_t vocabulary{0};

  bool Ids(std::span<const std::uint32_t> ids, std::string* error) {
    for (std::uint32_t id : ids)
      if (id >= vocabulary)
        return Fail(error, "token id out of range");
    std::vector<int> host(ids.begin(), ids.end());
    return engine.Upload(engine.ids, host.data(), host.size() * 4, error) &&
           hidden.Reserve(ids.size() * kTextWidth * 4, error);
  }
};

TextEncoder::TextEncoder(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
TextEncoder::~TextEncoder() = default;

std::size_t TextEncoder::weight_bytes() const {
  return impl_->weight_bytes;
}

std::unique_ptr<TextEncoder> TextEncoder::Load(
    const std::filesystem::path& safetensors, std::string* error) {
  auto impl = std::make_unique<Impl>();
  if (!impl->engine.Init(error))
    return nullptr;
  SafetensorsFile file;
  if (!file.Open(safetensors, impl->engine.stream, error))
    return nullptr;
  const auto* embed = file.Find("model.embed_tokens.weight");
  if (embed == nullptr || embed->shape.size() != 2 ||
      embed->shape[1] != kTextWidth) {
    Fail(error, "not a Qwen3-0.6B text encoder: " + safetensors.string());
    return nullptr;
  }
  impl->vocabulary = embed->shape[0];
  if (!file.Load("model.embed_tokens.weight", embed->count(), impl->embed, true,
                 error) ||
      !LoadEncoderLayers(file, "model.", kTextLayers, kTextShape, impl->layers,
                         error) ||
      !file.Load("model.norm.weight", kTextWidth, impl->norm, true, error) ||
      !file.Finish(error))
    return nullptr;
  impl->weight_bytes = file.uploaded_bytes();
  return std::unique_ptr<TextEncoder>(new TextEncoder(std::move(impl)));
}

bool TextEncoder::Encode(std::span<const std::uint32_t> ids,
                         std::vector<float>* states, std::string* error) {
  Impl& m = *impl_;
  const int rows = static_cast<int>(ids.size());
  if (rows == 0)
    return Fail(error, "empty text");
  if (!m.Ids(ids, error))
    return false;
  Engine& e = m.engine;
  LaunchGather(m.embed.bf16(), static_cast<const int*>(e.ids.get()), rows,
               kTextWidth, m.hidden.f32(), e.stream);
  if (!e.EncoderLayers(m.layers, kTextShape, m.hidden.f32(), rows, false, true,
                       error))
    return false;
  LaunchRmsNorm(m.hidden.f32(), m.norm.bf16(), nullptr, nullptr, rows,
                kTextWidth, nullptr, m.hidden.f32(), e.stream);
  states->resize(std::size_t(rows) * kTextWidth);
  return e.Download(m.hidden.get(), states->size() * 4, states->data(), error);
}

bool TextEncoder::Embed(std::span<const std::uint32_t> ids,
                        std::vector<float>* embeds, std::string* error) {
  Impl& m = *impl_;
  const int rows = static_cast<int>(ids.size());
  if (rows == 0)
    return Fail(error, "empty lyrics");
  if (!m.Ids(ids, error))
    return false;
  LaunchGather(m.embed.bf16(), static_cast<const int*>(m.engine.ids.get()),
               rows, kTextWidth, m.hidden.f32(), m.engine.stream);
  embeds->resize(std::size_t(rows) * kTextWidth);
  return m.engine.Download(m.hidden.get(), embeds->size() * 4, embeds->data(),
                           error);
}

// ----------------------------------------------------------------- DiT

struct TimeEmbedding {
  DeviceBuffer linear1, bias1, linear2, bias2, proj, proj_bias;
};

struct Dit::Impl {
  Engine engine;
  // Condition encoder.
  DeviceBuffer text_projector;  // [2048, 1024]
  DeviceBuffer lyric_embed, lyric_embed_bias, lyric_norm;
  std::vector<EncoderLayer> lyric_layers;
  DeviceBuffer timbre_embed, timbre_embed_bias, timbre_norm, timbre_token;
  std::vector<EncoderLayer> timbre_layers;
  // Decoder.
  DeviceBuffer proj_in, proj_in_bias;  // [2560, 384]
  TimeEmbedding time_t, time_r;
  DeviceBuffer condition, condition_bias;  // [2560, 2048]
  std::vector<DitLayer> layers;
  DeviceBuffer norm_out, out_table;  // table BF16 [2, 2560]
  DeviceBuffer proj_out;             // transposed [128, 2560]
  std::vector<float> proj_out_bias;  // [64]
  // Audio-code path: FSQ projection (host) and detokenizer.
  std::vector<float> quant_out, quant_out_bias;  // [2048, 6], [2048]
  DeviceBuffer detok_embed, detok_embed_bias, detok_special, detok_norm,
      detok_proj, detok_proj_bias;
  std::vector<EncoderLayer> detok_layers;
  // Audio tokenizer: acoustic projection, attention pooler, FSQ project_in.
  DeviceBuffer tok_proj, tok_proj_bias;  // [2048, 64]
  DeviceBuffer pool_embed, pool_embed_bias, pool_token, pool_norm;
  std::vector<EncoderLayer> pool_layers;
  std::vector<float> quant_in, quant_in_bias;  // [6, 2048], [6]
  // Null condition for classifier-free guidance, BF16 [1, 2048].
  DeviceBuffer null_emb;
  // Activations.
  DeviceBuffer staging, input, hidden, sinusoid, time_hidden, temb, proj,
      temb_r, proj_r, modulation, temb2, out_mod, output;
  DeviceBuffer cond_bf16, cond_states, cross_kv;
  int condition_rows{0};
  std::vector<float> context;  // [frames, 128]
  int frames{0};
  hipEvent_t timing[2]{};
  double velocity_ms{0};
  std::size_t weight_bytes{0};

  ~Impl() {
    for (hipEvent_t event : timing)
      if (event != nullptr)
        (void)hipEventDestroy(event);
  }

  bool Time(const TimeEmbedding& te, float t, float* temb_out,
            float* proj_out_values, std::string* error) {
    float frequencies[kFrequencies];
    for (int i = 0; i < kFrequencies / 2; ++i) {
      const float freq = std::exp(-std::log(10000.0F) * static_cast<float>(i) /
                                  (kFrequencies / 2));
      const float arg = t * 1000.0F * freq;
      frequencies[i] = std::cos(arg);
      frequencies[kFrequencies / 2 + i] = std::sin(arg);
    }
    Engine& e = engine;
    if (!e.Upload(sinusoid, frequencies, sizeof(frequencies), error))
      return false;
    const int w = kDitShape.width;
    LaunchMatVec(te.linear1.bf16(), te.bias1.bf16(), sinusoid.f32(),
                 kFrequencies, w, false, time_hidden.f32(), e.stream);
    LaunchMatVec(te.linear2.bf16(), te.bias2.bf16(), time_hidden.f32(), w, w,
                 true, temb_out, e.stream);
    LaunchMatVec(te.proj.bf16(), te.proj_bias.bf16(), temb_out, w, 6 * w, true,
                 proj_out_values, e.stream);
    // The sinusoid buffer is reused by the next call: wait for its readers.
    return Check(hipStreamSynchronize(e.stream), "time embedding", error);
  }

  // Projects packed condition states (BF16 [rows, 2048]) through the
  // condition embedder and every layer's cross-attention key / value
  // projections into the layers' caches (`null` selects the null set).
  bool ProjectConditions(const DeviceBuffer& states, int rows, bool null,
                         std::string* error) {
    Engine& e = engine;
    const Shape& s = kDitShape;
    if (!cond_states.Reserve(std::size_t(rows) * s.width * 2, error) ||
        !cross_kv.Reserve(std::size_t(rows) * 2 * s.kv_dim() * 2, error) ||
        !e.value.Reserve(std::size_t(rows) * s.kv_dim() * 2, error) ||
        !e.Gemm(condition, &condition_bias, states.bf16(), cond_states.bf16(),
                rows, s.width, kConditionWidth, error))
      return false;
    for (DitLayer& l : layers) {
      DeviceBuffer& key = null ? l.null_key : l.cross_key;
      DeviceBuffer& packed = null ? l.null_packed : l.cross_packed;
      if (!key.Reserve(std::size_t(rows) * s.kv_dim() * 2, error) ||
          !packed.Reserve(std::size_t(PaddedRows(rows)) * s.kv_dim() * 2,
                          error) ||
          !e.Gemm(l.cross_attn.kv, nullptr, cond_states.bf16(), cross_kv.bf16(),
                  rows, 2 * s.kv_dim(), s.width, error))
        return false;
      LaunchHeads(cross_kv.bf16(), 2 * s.kv_dim(), 0, rows, s.kv_heads,
                  l.cross_attn.k_norm.bf16(), false, key.bf16(), e.stream);
      LaunchHeads(cross_kv.bf16(), 2 * s.kv_dim(), s.kv_dim(), rows,
                  s.kv_heads, nullptr, false, e.value.bf16(), e.stream);
      LaunchPackValues(e.value.bf16(), packed.bf16(), rows, s.kv_heads,
                       e.stream);
    }
    return Check(hipStreamSynchronize(e.stream), "conditions", error);
  }
};

Dit::Dit(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Dit::~Dit() = default;

std::size_t Dit::weight_bytes() const {
  return impl_->weight_bytes;
}
double Dit::last_velocity_ms() const {
  return impl_->velocity_ms;
}

std::unique_ptr<Dit> Dit::Load(const std::filesystem::path& safetensors,
                               std::string* error) {
  auto impl = std::make_unique<Impl>();
  Impl& m = *impl;
  if (!m.engine.Init(error))
    return nullptr;
  for (hipEvent_t& event : m.timing)
    if (!Check(hipEventCreate(&event), "event", error))
      return nullptr;
  SafetensorsFile file;
  if (!file.Open(safetensors, m.engine.stream, error))
    return nullptr;
  const auto* condition = file.Find("decoder.condition_embedder.weight");
  if (!file.Has("decoder.layers.31.mlp.down_proj.weight") ||
      condition == nullptr ||
      condition->shape != std::vector<std::size_t>{2560, 2048}) {
    Fail(error, "not an ACE-Step 1.5 XL DiT: " + safetensors.string());
    return nullptr;
  }
  const int ew = kEncoderShape.width, dw = kDitShape.width;
  auto load = [&](const std::string& name, std::size_t count,
                  DeviceBuffer& buffer) {
    return file.Load(name, count, buffer, true, error);
  };
  bool ok =
      load("encoder.text_projector.weight", std::size_t(ew) * kTextWidth,
           m.text_projector) &&
      load("encoder.lyric_encoder.embed_tokens.weight",
           std::size_t(ew) * kTextWidth, m.lyric_embed) &&
      load("encoder.lyric_encoder.embed_tokens.bias", ew, m.lyric_embed_bias) &&
      load("encoder.lyric_encoder.norm.weight", ew, m.lyric_norm) &&
      LoadEncoderLayers(file, "encoder.lyric_encoder.", kLyricLayers,
                        kEncoderShape, m.lyric_layers, error) &&
      load("encoder.timbre_encoder.embed_tokens.weight",
           std::size_t(ew) * kLatentChannels, m.timbre_embed) &&
      load("encoder.timbre_encoder.embed_tokens.bias", ew,
           m.timbre_embed_bias) &&
      load("encoder.timbre_encoder.norm.weight", ew, m.timbre_norm) &&
      file.Load("encoder.timbre_encoder.special_token", ew, m.timbre_token,
                false, error) &&
      LoadEncoderLayers(file, "encoder.timbre_encoder.", kTimbreLayers,
                        kEncoderShape, m.timbre_layers, error) &&
      load("decoder.proj_in.1.weight", std::size_t(dw) * 384, m.proj_in) &&
      load("decoder.proj_in.1.bias", dw, m.proj_in_bias) &&
      load("decoder.condition_embedder.weight", std::size_t(dw) * ew,
           m.condition) &&
      load("decoder.condition_embedder.bias", dw, m.condition_bias) &&
      load("decoder.norm_out.weight", dw, m.norm_out) &&
      load("decoder.scale_shift_table", 2 * std::size_t(dw), m.out_table);
  for (auto [prefix, te] : {std::pair{"decoder.time_embed.", &m.time_t},
                            std::pair{"decoder.time_embed_r.", &m.time_r}}) {
    const std::string p = prefix;
    ok = ok &&
         load(p + "linear_1.weight", std::size_t(dw) * kFrequencies,
              te->linear1) &&
         load(p + "linear_1.bias", dw, te->bias1) &&
         load(p + "linear_2.weight", std::size_t(dw) * dw, te->linear2) &&
         load(p + "linear_2.bias", dw, te->bias2) &&
         load(p + "time_proj.weight", std::size_t(6) * dw * dw, te->proj) &&
         load(p + "time_proj.bias", 6 * std::size_t(dw), te->proj_bias);
  }
  m.layers.resize(kDitLayers);
  for (int i = 0; ok && i < kDitLayers; ++i) {
    const std::string p = "decoder.layers." + std::to_string(i) + ".";
    DitLayer& l = m.layers[i];
    ok = load(p + "self_attn_norm.weight", dw, l.self_norm) &&
         load(p + "cross_attn_norm.weight", dw, l.cross_norm) &&
         load(p + "mlp_norm.weight", dw, l.mlp_norm) &&
         load(p + "scale_shift_table", 6 * std::size_t(dw), l.table) &&
         LoadAttention(file, p + "self_attn.", kDitShape, false, l.self_attn,
                       error) &&
         LoadAttention(file, p + "cross_attn.", kDitShape, true, l.cross_attn,
                       error) &&
         LoadMlp(file, p + "mlp.", kDitShape, l.mlp, error);
  }
  ok = ok &&
       file.ReadHost("tokenizer.quantizer.project_out.weight",
                     std::size_t(ew) * 6, &m.quant_out, error) &&
       file.ReadHost("tokenizer.quantizer.project_out.bias", ew,
                     &m.quant_out_bias, error) &&
       load("detokenizer.embed_tokens.weight", std::size_t(ew) * ew,
            m.detok_embed) &&
       load("detokenizer.embed_tokens.bias", ew, m.detok_embed_bias) &&
       file.Load("detokenizer.special_tokens", 5 * std::size_t(ew),
                 m.detok_special, false, error) &&
       LoadEncoderLayers(file, "detokenizer.", 2, kEncoderShape, m.detok_layers,
                         error) &&
       load("detokenizer.norm.weight", ew, m.detok_norm) &&
       load("detokenizer.proj_out.weight", std::size_t(kLatentChannels) * ew,
            m.detok_proj) &&
       load("detokenizer.proj_out.bias", kLatentChannels, m.detok_proj_bias);
  // Audio tokenizer (cover tasks) and the null condition (CFG).
  ok = ok &&
       load("tokenizer.audio_acoustic_proj.weight",
            std::size_t(ew) * kLatentChannels, m.tok_proj) &&
       load("tokenizer.audio_acoustic_proj.bias", ew, m.tok_proj_bias) &&
       load("tokenizer.attention_pooler.embed_tokens.weight",
            std::size_t(ew) * ew, m.pool_embed) &&
       load("tokenizer.attention_pooler.embed_tokens.bias", ew,
            m.pool_embed_bias) &&
       file.Load("tokenizer.attention_pooler.special_token", ew, m.pool_token,
                 false, error) &&
       LoadEncoderLayers(file, "tokenizer.attention_pooler.", 2, kEncoderShape,
                         m.pool_layers, error) &&
       load("tokenizer.attention_pooler.norm.weight", ew, m.pool_norm) &&
       file.ReadHost("tokenizer.quantizer.project_in.weight",
                     std::size_t(6) * ew, &m.quant_in, error) &&
       file.ReadHost("tokenizer.quantizer.project_in.bias", 6,
                     &m.quant_in_bias, error) &&
       load("null_condition_emb", ew, m.null_emb);
  // ConvTranspose1d weight [2560 in, 64 out, 2 taps] -> GEMM rows o * 2 + k.
  std::vector<float> out_weight;
  ok = ok &&
       file.ReadHost("decoder.proj_out.1.weight", std::size_t(dw) * 128,
                     &out_weight, error) &&
       file.ReadHost("decoder.proj_out.1.bias", kLatentChannels,
                     &m.proj_out_bias, error);
  if (ok) {
    std::vector<std::uint16_t> transposed(std::size_t(dw) * 128);
    for (int c = 0; c < dw; ++c)
      for (int r = 0; r < 128; ++r)
        transposed[std::size_t(r) * dw + c] =
            HostBf16(out_weight[std::size_t(c) * 128 + r]);
    ok = m.proj_out.Allocate(transposed.size() * 2, error) &&
         Check(hipMemcpy(m.proj_out.get(), transposed.data(),
                         transposed.size() * 2, hipMemcpyHostToDevice),
               "upload", error);
  }
  ok = ok && file.Finish(error);
  if (!ok)
    return nullptr;
  m.weight_bytes = file.uploaded_bytes() + std::size_t(dw) * 256;
  const std::size_t w = dw;
  if (!m.time_hidden.Allocate(w * 4, error) || !m.temb.Allocate(w * 4, error) ||
      !m.proj.Allocate(6 * w * 4, error) || !m.temb_r.Allocate(w * 4, error) ||
      !m.proj_r.Allocate(6 * w * 4, error) ||
      !m.modulation.Allocate(6 * w * 4, error) ||
      !m.temb2.Allocate(2 * w * 4, error) ||
      !m.out_mod.Allocate(2 * w * 4, error))
    return nullptr;
  // The null condition never changes: its keys / values are cached once.
  // All rows of the official null batch are the same vector, so one row
  // attends identically.
  if (!m.ProjectConditions(m.null_emb, 1, true, error))
    return nullptr;
  return std::unique_ptr<Dit>(new Dit(std::move(impl)));
}

bool Dit::EncodeConditions(std::span<const float> text, int text_rows,
                           std::span<const float> lyric, int lyric_rows,
                           std::span<const float> timbre, int timbre_rows,
                           std::vector<float>* conditions, std::string* error) {
  Impl& m = *impl_;
  Engine& e = m.engine;
  const int ew = kEncoderShape.width;
  if (text_rows <= 0 || lyric_rows <= 0 || timbre_rows <= 0 ||
      text.size() != std::size_t(text_rows) * kTextWidth ||
      lyric.size() != std::size_t(lyric_rows) * kTextWidth ||
      timbre.size() != std::size_t(timbre_rows) * kLatentChannels)
    return Fail(error, "ACE-Step condition inputs have wrong sizes");
  const int rows = lyric_rows + 1 + text_rows;
  conditions->resize(std::size_t(rows) * ew);
  const int most = std::max({text_rows, lyric_rows, timbre_rows + 1});
  if (!e.Reserve(most, kEncoderShape, error) ||
      !m.hidden.Reserve(std::size_t(most) * ew * 4, error) ||
      !m.output.Reserve(std::size_t(most) * ew * 4, error))
    return false;
  float* out = conditions->data();
  // Text: projector only.
  if (!e.UploadBf16(text, m.staging, m.input, error) ||
      !e.Gemm(m.text_projector, nullptr, m.input.bf16(), e.proj.bf16(),
              text_rows, ew, kTextWidth, error))
    return false;
  LaunchToF32(e.proj.bf16(), std::size_t(text_rows) * ew, m.output.f32(),
              e.stream);
  if (!e.Download(m.output.get(), std::size_t(text_rows) * ew * 4,
                  out + std::size_t(lyric_rows + 1) * ew, error))
    return false;
  // Lyrics: embedding projection, 8 windowed layers, norm.
  if (!e.UploadBf16(lyric, m.staging, m.input, error) ||
      !e.Gemm(m.lyric_embed, &m.lyric_embed_bias, m.input.bf16(), e.proj.bf16(),
              lyric_rows, ew, kTextWidth, error))
    return false;
  LaunchToF32(e.proj.bf16(), std::size_t(lyric_rows) * ew, m.hidden.f32(),
              e.stream);
  if (!e.EncoderLayers(m.lyric_layers, kEncoderShape, m.hidden.f32(),
                       lyric_rows, true, false, error))
    return false;
  LaunchRmsNorm(m.hidden.f32(), m.lyric_norm.bf16(), nullptr, nullptr,
                lyric_rows, ew, nullptr, m.output.f32(), e.stream);
  if (!e.Download(m.output.get(), std::size_t(lyric_rows) * ew * 4, out, error))
    return false;
  // Timbre: CLS token + projected latents, 4 windowed layers, norm, row 0.
  if (!e.UploadBf16(timbre, m.staging, m.input, error) ||
      !e.Gemm(m.timbre_embed, &m.timbre_embed_bias, m.input.bf16(),
              e.proj.bf16(), timbre_rows, ew, kLatentChannels, error) ||
      !Check(hipMemcpyAsync(m.hidden.get(), m.timbre_token.get(), ew * 4,
                            hipMemcpyDeviceToDevice, e.stream),
             "copy", error))
    return false;
  LaunchToF32(e.proj.bf16(), std::size_t(timbre_rows) * ew, m.hidden.f32() + ew,
              e.stream);
  if (!e.EncoderLayers(m.timbre_layers, kEncoderShape, m.hidden.f32(),
                       timbre_rows + 1, true, false, error))
    return false;
  LaunchRmsNorm(m.hidden.f32(), m.timbre_norm.bf16(), nullptr, nullptr, 1, ew,
                nullptr, m.output.f32(), e.stream);
  return e.Download(m.output.get(), std::size_t(ew) * 4,
                    out + std::size_t(lyric_rows) * ew, error);
}

bool Dit::SetConditions(std::span<const float> conditions, int rows,
                        std::span<const float> context, int frames,
                        std::string* error) {
  Impl& m = *impl_;
  Engine& e = m.engine;
  if (rows <= 0 || frames <= 0 ||
      conditions.size() != std::size_t(rows) * kConditionWidth ||
      context.size() != std::size_t(frames) * 2 * kLatentChannels)
    return Fail(error, "ACE-Step DiT conditions have wrong sizes");
  if (!e.UploadBf16(conditions, m.staging, m.cond_bf16, error) ||
      !m.ProjectConditions(m.cond_bf16, rows, false, error))
    return false;
  m.condition_rows = rows;
  m.context.assign(context.begin(), context.end());
  m.frames = frames;
  return true;
}

bool Dit::Velocity(std::span<const float> xt, float t, std::vector<float>* v,
                   DitProbes* probes, std::string* error, bool unconditional) {
  Impl& m = *impl_;
  Engine& e = m.engine;
  const Shape& s = kDitShape;
  const int w = s.width;
  const int frames = m.frames;
  if (frames == 0 || m.condition_rows == 0)
    return Fail(error, "ACE-Step DiT conditions are not set");
  if (xt.size() != std::size_t(frames) * kLatentChannels)
    return Fail(error, "ACE-Step latent has the wrong size");
  const int tokens = (frames + 1) / 2;
  if (!e.Reserve(tokens, s, error) ||
      !m.hidden.Reserve(std::size_t(tokens) * w * 4, error) ||
      !m.output.Reserve(std::size_t(tokens) * 128 * 2, error))
    return false;
  (void)hipEventRecord(m.timing[0], e.stream);
  // Timestep embeddings of t and t - r = 0.
  if (!m.Time(m.time_t, t, m.temb.f32(), m.proj.f32(), error) ||
      !m.Time(m.time_r, 0.0F, m.temb_r.f32(), m.proj_r.f32(), error))
    return false;
  LaunchAdd(m.temb.f32(), m.temb_r.f32(), w, e.stream);
  LaunchAdd(m.proj.f32(), m.proj_r.f32(), 6 * std::size_t(w), e.stream);
  // Patches [tokens, 192 channels x 2 taps] of [context | x_t], zero padded.
  std::vector<std::uint16_t> patches(std::size_t(tokens) * 384, 0);
  for (int f = 0; f < frames; ++f) {
    const int j = f / 2, k = f % 2;
    for (int c = 0; c < 128; ++c)
      patches[std::size_t(j) * 384 + c * 2 + k] =
          HostBf16(m.context[std::size_t(f) * 128 + c]);
    for (int c = 0; c < kLatentChannels; ++c)
      patches[std::size_t(j) * 384 + (128 + c) * 2 + k] =
          HostBf16(xt[std::size_t(f) * kLatentChannels + c]);
  }
  if (!e.Upload(m.input, patches.data(), patches.size() * 2, error) ||
      !e.Gemm(m.proj_in, &m.proj_in_bias, m.input.bf16(), e.proj.bf16(), tokens,
              w, 384, error))
    return false;
  LaunchToF32(e.proj.bf16(), std::size_t(tokens) * w, m.hidden.f32(), e.stream);
  auto probe = [&](std::vector<float>* target) {
    if (target == nullptr)
      return true;
    target->resize(std::size_t(tokens) * w);
    return e.Download(m.hidden.get(), target->size() * 4, target->data(),
                      error);
  };
  if (probes != nullptr && !probe(&probes->patch))
    return false;
  float* mod = m.modulation.f32();
  for (int i = 0; i < kDitLayers; ++i) {
    const DitLayer& l = m.layers[i];
    LaunchAddTable(l.table.bf16(), m.proj.f32(), 6 * std::size_t(w), mod,
                   e.stream);
    // Self-attention with AdaLN (shift 0, scale 1, gate 2).
    LaunchRmsNorm(m.hidden.f32(), l.self_norm.bf16(), mod + w, mod, tokens, w,
                  e.normed.bf16(), nullptr, e.stream);
    AttentionMask mask;
    mask.window = i % 2 == 0 ? kWindow : 0;
    if (!e.SelfAttention(l.self_attn, s, tokens, mask, error))
      return false;
    LaunchResidual(m.hidden.f32(), e.proj.bf16(), mod + 2 * w, tokens, w,
                   e.stream);
    // Cross-attention to the cached condition keys / values.
    LaunchRmsNorm(m.hidden.f32(), l.cross_norm.bf16(), nullptr, nullptr, tokens,
                  w, e.normed.bf16(), nullptr, e.stream);
    if (!e.Gemm(l.cross_attn.q, nullptr, e.normed.bf16(), e.qkv.bf16(), tokens,
                s.q_dim(), w, error))
      return false;
    LaunchHeads(e.qkv.bf16(), s.q_dim(), 0, tokens, s.heads,
                l.cross_attn.q_norm.bf16(), false, e.query.bf16(), e.stream);
    LaunchAttention(e.query.bf16(),
                    unconditional ? l.null_key.bf16() : l.cross_key.bf16(),
                    unconditional ? l.null_packed.bf16()
                                  : l.cross_packed.bf16(),
                    e.attn.bf16(), tokens, unconditional ? 1 : m.condition_rows,
                    s.heads, s.kv_heads, AttentionMask{}, e.stream);
    if (!e.Gemm(l.cross_attn.out, nullptr, e.attn.bf16(), e.proj.bf16(), tokens,
                w, s.q_dim(), error))
      return false;
    LaunchResidual(m.hidden.f32(), e.proj.bf16(), nullptr, tokens, w, e.stream);
    // Feed-forward with AdaLN (shift 3, scale 4, gate 5).
    LaunchRmsNorm(m.hidden.f32(), l.mlp_norm.bf16(), mod + 4 * w, mod + 3 * w,
                  tokens, w, e.normed.bf16(), nullptr, e.stream);
    if (!e.FeedForward(l.mlp, s, tokens, error))
      return false;
    LaunchResidual(m.hidden.f32(), e.proj.bf16(), mod + 5 * w, tokens, w,
                   e.stream);
    if (probes != nullptr && i == 0 && !probe(&probes->layer0))
      return false;
    if (probes != nullptr && i == kDitLayers - 1 && !probe(&probes->layer31))
      return false;
  }
  // Output: shift, scale = table + temb; norm; transposed-conv projection.
  for (int r = 0; r < 2; ++r)
    if (!Check(hipMemcpyAsync(m.temb2.f32() + std::size_t(r) * w, m.temb.f32(),
                              std::size_t(w) * 4, hipMemcpyDeviceToDevice,
                              e.stream),
               "copy", error))
      return false;
  LaunchAddTable(m.out_table.bf16(), m.temb2.f32(), 2 * std::size_t(w),
                 m.out_mod.f32(), e.stream);
  LaunchRmsNorm(m.hidden.f32(), m.norm_out.bf16(), m.out_mod.f32() + w,
                m.out_mod.f32(), tokens, w, e.normed.bf16(), nullptr, e.stream);
  if (!e.Gemm(m.proj_out, nullptr, e.normed.bf16(), m.output.bf16(), tokens,
              128, w, error))
    return false;
  (void)hipEventRecord(m.timing[1], e.stream);
  std::vector<std::uint16_t> y(std::size_t(tokens) * 128);
  if (!e.Download(m.output.get(), y.size() * 2, y.data(), error))
    return false;
  float ms = 0;
  if (hipEventElapsedTime(&ms, m.timing[0], m.timing[1]) == hipSuccess)
    m.velocity_ms = ms;
  v->resize(std::size_t(frames) * kLatentChannels);
  for (int f = 0; f < frames; ++f) {
    const int j = f / 2, k = f % 2;
    for (int o = 0; o < kLatentChannels; ++o)
      (*v)[std::size_t(f) * kLatentChannels + o] =
          HostFloat(y[std::size_t(j) * 128 + o * 2 + k]) + m.proj_out_bias[o];
  }
  return true;
}

bool Dit::CodeHints(std::span<const int> codes, int frames,
                    std::vector<float>* hints, std::string* error) {
  Impl& m = *impl_;
  Engine& e = m.engine;
  const int ew = kEncoderShape.width;
  constexpr int kPool = 5;
  constexpr int kLevels[6] = {8, 8, 8, 5, 5, 5};
  const int count =
      std::max(static_cast<int>(codes.size()), (frames + kPool - 1) / kPool);
  if (count == 0)
    return Fail(error, "no audio codes");
  // FSQ: index -> per-level digit -> [-1, 1] code, then project_out.
  std::vector<float> projected(std::size_t(count) * ew);
  for (int n = 0; n < count; ++n) {
    int index = n < static_cast<int>(codes.size()) ? codes[n] : 35847;
    if (index < 0 || index >= 64000)
      return Fail(error, "audio code out of range");
    float digits[6];
    for (int l = 0; l < 6; ++l) {
      digits[l] = static_cast<float>(index % kLevels[l]) *
                      (2.0F / static_cast<float>(kLevels[l] - 1)) -
                  1.0F;
      index /= kLevels[l];
    }
    for (int r = 0; r < ew; ++r) {
      float sum = m.quant_out_bias[r];
      for (int l = 0; l < 6; ++l)
        sum += m.quant_out[std::size_t(r) * 6 + l] * digits[l];
      projected[std::size_t(n) * ew + r] = sum;
    }
  }
  const int rows = count * kPool;
  if (!e.Reserve(rows, kEncoderShape, error) ||
      !m.hidden.Reserve(std::size_t(rows) * ew * 4, error) ||
      !m.output.Reserve(std::size_t(rows) * ew * 4, error) ||
      !e.UploadBf16(projected, m.staging, m.input, error) ||
      !e.Gemm(m.detok_embed, &m.detok_embed_bias, m.input.bf16(), e.proj.bf16(),
              count, ew, ew, error))
    return false;
  LaunchToF32(e.proj.bf16(), std::size_t(count) * ew, m.output.f32(), e.stream);
  LaunchExpand(m.output.f32(), m.detok_special.f32(), count, kPool, ew,
               m.hidden.f32(), e.stream);
  if (!e.EncoderLayers(m.detok_layers, kEncoderShape, m.hidden.f32(), rows,
                       false, false, error, kPool))
    return false;
  LaunchRmsNorm(m.hidden.f32(), m.detok_norm.bf16(), nullptr, nullptr, rows, ew,
                e.normed.bf16(), nullptr, e.stream);
  if (!e.Gemm(m.detok_proj, &m.detok_proj_bias, e.normed.bf16(), e.proj.bf16(),
              rows, kLatentChannels, ew, error))
    return false;
  std::vector<std::uint16_t> y(std::size_t(rows) * kLatentChannels);
  if (!e.Download(e.proj.get(), y.size() * 2, y.data(), error))
    return false;
  hints->resize(std::size_t(frames) * kLatentChannels);
  for (std::size_t i = 0; i < hints->size(); ++i)
    (*hints)[i] = HostFloat(y[i]);
  return true;
}

bool Dit::Tokenize(std::span<const float> latents, int frames,
                   std::span<const float> silence, std::vector<int>* codes,
                   std::string* error) {
  Impl& m = *impl_;
  Engine& e = m.engine;
  const int ew = kEncoderShape.width;
  constexpr int kPool = 5;
  constexpr int kLevels[6] = {8, 8, 8, 5, 5, 5};
  if (frames <= 0 || latents.size() < std::size_t(frames) * kLatentChannels)
    return Fail(error, "ACE-Step tokenizer input has the wrong size");
  const int groups = (frames + kPool - 1) / kPool;
  const int padded = groups * kPool;
  if (padded > frames &&
      silence.size() < std::size_t(padded - frames) * kLatentChannels)
    return Fail(error, "ACE-Step tokenizer needs silence frames for padding");
  // Padded latents [padded, 64]: the official `tokenize` appends the first
  // frames of the silence latent.
  std::vector<float> input(std::size_t(padded) * kLatentChannels);
  std::copy_n(latents.begin(), std::size_t(frames) * kLatentChannels,
              input.begin());
  for (int f = frames; f < padded; ++f)
    std::copy_n(silence.begin() + std::size_t(f - frames) * kLatentChannels,
                kLatentChannels,
                input.begin() + std::size_t(f) * kLatentChannels);
  codes->assign(std::size_t(groups), 0);
  // Groups are independent sequences: work in batches of them.
  constexpr int kBatchGroups = 256;
  std::vector<float> pooled(std::size_t(kBatchGroups) * ew);
  for (int g0 = 0; g0 < groups; g0 += kBatchGroups) {
    const int count = std::min(kBatchGroups, groups - g0);
    const int rows5 = count * kPool, rows6 = count * (kPool + 1);
    const std::span<const float> slice(
        input.data() + std::size_t(g0) * kPool * kLatentChannels,
        std::size_t(rows5) * kLatentChannels);
    if (!e.Reserve(rows6, kEncoderShape, error) ||
        !m.hidden.Reserve(std::size_t(rows6) * ew * 4, error) ||
        !m.output.Reserve(std::size_t(rows6) * ew * 4, error) ||
        !e.UploadBf16(slice, m.staging, m.input, error) ||
        // audio_acoustic_proj: [rows5, 64] -> [rows5, 2048]
        !e.Gemm(m.tok_proj, &m.tok_proj_bias, m.input.bf16(), e.proj.bf16(),
                rows5, ew, kLatentChannels, error) ||
        // attention_pooler.embed_tokens: [rows5, 2048] -> [rows5, 2048]
        !e.Gemm(m.pool_embed, &m.pool_embed_bias, e.proj.bf16(), e.qkv.bf16(),
                rows5, ew, ew, error))
      return false;
    LaunchToF32(e.qkv.bf16(), std::size_t(rows5) * ew, m.output.f32(),
                e.stream);
    LaunchGroupPrepend(m.output.f32(), m.pool_token.f32(), count, kPool, ew,
                       m.hidden.f32(), e.stream);
    if (!e.EncoderLayers(m.pool_layers, kEncoderShape, m.hidden.f32(), rows6,
                         true, false, error, kPool + 1))
      return false;
    LaunchRmsNorm(m.hidden.f32(), m.pool_norm.bf16(), nullptr, nullptr, rows6,
                  ew, nullptr, m.output.f32(), e.stream);
    // Row 0 of every group is the pooled representation.
    if (!Check(hipMemcpy2DAsync(pooled.data(), std::size_t(ew) * 4,
                                m.output.get(),
                                std::size_t(kPool + 1) * ew * 4,
                                std::size_t(ew) * 4, count,
                                hipMemcpyDeviceToHost, e.stream),
               "download", error) ||
        !Check(hipStreamSynchronize(e.stream), "tokenizer", error))
      return false;
    // ResidualFSQ (one quantizer) on the host in FP32: project_in, soft
    // clamp tanh(x / v) * v with v = 1 + 1 / (L - 1), then the bound
    // floor((L - 1) * (tanh(z) + 1) / 2 + 0.5) and the index.
    for (int g = 0; g < count; ++g) {
      const float* row = pooled.data() + std::size_t(g) * ew;
      int index = 0, basis = 1;
      for (int l = 0; l < 6; ++l) {
        double z = m.quant_in_bias[std::size_t(l)];
        for (int c = 0; c < ew; ++c)
          z += double(m.quant_in[std::size_t(l) * ew + c]) * double(row[c]);
        const double levels = kLevels[l];
        const double clamp = 1.0 + 1.0 / (levels - 1.0);
        z = std::tanh(z / clamp) * clamp;
        const double bracket =
            (levels - 1.0) * (std::tanh(z) + 1.0) / 2.0 + 0.5;
        const int digit = std::clamp(static_cast<int>(std::floor(bracket)), 0,
                                     kLevels[l] - 1);
        index += digit * basis;
        basis *= kLevels[l];
      }
      (*codes)[std::size_t(g0 + g)] = index;
    }
  }
  return true;
}

// ----------------------------------------------------------------- code LM

namespace {

struct LmShape {
  Shape shape{2560, 32, 8, 9728};  // Qwen3-4B
  int layers{36};
};

struct KvCache {
  DeviceBuffer key, value;  // all layers, concatenated
  int capacity{0};
  int length{0};
};

}  // namespace

struct CodeLm::Impl {
  Engine engine;
  LmShape lm;          // read from the weights
  DeviceBuffer embed;  // [vocabulary, width]
  DeviceBuffer head;   // untied lm_head [vocabulary, width], else empty
  std::vector<EncoderLayer> layers;
  DeviceBuffer norm;
  DeviceBuffer hidden, positions, logits;
  KvCache cache[2];
  std::size_t vocabulary{0};
  std::size_t weight_bytes{0};
  std::vector<float> first_logits;
  double codes_per_second{0};

  std::size_t LayerKeys(const KvCache& c) const {
    return std::size_t(lm.shape.kv_heads) * c.capacity * kHeadDim;
  }
  std::size_t LayerValues(const KvCache& c) const {
    return std::size_t(lm.shape.kv_heads) * PaddedRows(c.capacity) * kHeadDim;
  }

  bool Allocate(KvCache& c, int capacity, std::string* error) {
    c.capacity = capacity;
    c.length = 0;
    const std::size_t keys = LayerKeys(c) * lm.layers * 2;
    const std::size_t values = LayerValues(c) * lm.layers * 2;
    return c.key.Reserve(keys, error) && c.value.Reserve(values, error) &&
           Check(hipMemsetAsync(c.key.get(), 0, keys, engine.stream), "memset",
                 error) &&
           Check(hipMemsetAsync(c.value.get(), 0, values, engine.stream),
                 "memset", error);
  }

  // Runs `rows` tokens (hidden rows [first, first + rows), RoPE positions in
  // `positions`) of sequence `seq` through layer `i`; they continue the
  // cached sequence.
  bool Layer(int i, int seq, int first, int rows, bool causal,
             std::string* error) {
    Engine& e = engine;
    const Shape& s = lm.shape;
    const EncoderLayer& l = layers[i];
    KvCache& c = cache[seq];
    float* x = hidden.f32() + std::size_t(first) * s.width;
    const int* pos = static_cast<const int*>(positions.get()) + first;
    std::uint16_t* key_cache = c.key.bf16() + LayerKeys(c) * i;
    std::uint16_t* value_cache = c.value.bf16() + LayerValues(c) * i;
    const int stride = s.q_dim() + 2 * s.kv_dim();
    LaunchRmsNorm(x, l.input_norm.bf16(), nullptr, nullptr, rows, s.width,
                  e.normed.bf16(), nullptr, e.stream);
    if (!e.Gemm(l.attn.qkv, nullptr, e.normed.bf16(), e.qkv.bf16(), rows,
                stride, s.width, error))
      return false;
    LaunchHeads(e.qkv.bf16(), stride, 0, rows, s.heads, l.attn.q_norm.bf16(),
                true, e.query.bf16(), e.stream, pos);
    LaunchHeads(e.qkv.bf16(), stride, s.q_dim(), rows, s.kv_heads,
                l.attn.k_norm.bf16(), true, e.key.bf16(), e.stream, pos);
    LaunchHeads(e.qkv.bf16(), stride, s.q_dim() + s.kv_dim(), rows, s.kv_heads,
                nullptr, false, e.value.bf16(), e.stream);
    LaunchStoreCache(e.key.bf16(), e.value.bf16(), rows, s.kv_heads, c.length,
                     c.capacity, key_cache, value_cache, e.stream);
    AttentionMask mask;
    mask.causal = causal;
    mask.capacity = static_cast<std::uint32_t>(c.capacity);
    LaunchAttention(e.query.bf16(), key_cache, value_cache, e.attn.bf16(), rows,
                    c.length + rows, s.heads, s.kv_heads, mask, e.stream);
    if (!e.Gemm(l.attn.out, nullptr, e.attn.bf16(), e.proj.bf16(), rows,
                s.width, s.q_dim(), error))
      return false;
    LaunchResidual(x, e.proj.bf16(), nullptr, rows, s.width, e.stream);
    LaunchRmsNorm(x, l.post_norm.bf16(), nullptr, nullptr, rows, s.width,
                  e.normed.bf16(), nullptr, e.stream);
    if (!e.FeedForward(l.mlp, s, rows, error))
      return false;
    LaunchResidual(x, e.proj.bf16(), nullptr, rows, s.width, e.stream);
    return true;
  }

  // Audio-code logits of hidden rows [0, n) after the final norm, to host
  // `out` [n, 64000].
  bool Logits(int n, std::vector<float>* out, std::string* error) {
    Engine& e = engine;
    const int w = lm.shape.width;
    LaunchRmsNorm(hidden.f32(), norm.bf16(), nullptr, nullptr, n, w,
                  e.normed.bf16(), nullptr, e.stream);
    const DeviceBuffer& table = head.get() != nullptr ? head : embed;
    if (!logits.Reserve(std::size_t(n) * kCodes * 2, error) ||
        !e.gemm.Run(table.bf16() + std::size_t(kFirstCode) * w, nullptr,
                    e.normed.bf16(), logits.bf16(), n, kCodes, w, error))
      return false;
    std::vector<std::uint16_t> bits(std::size_t(n) * kCodes);
    if (!e.Download(logits.get(), bits.size() * 2, bits.data(), error))
      return false;
    out->resize(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i)
      (*out)[i] = HostFloat(bits[i]);
    return true;
  }

  // Embeds `ids` into hidden rows from `first` with RoPE positions from
  // `start`.
  bool Embed(std::span<const std::uint32_t> ids, int first, int start,
             std::string* error) {
    std::vector<int> host(ids.begin(), ids.end()), pos(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
      if (ids[i] >= vocabulary)
        return Fail(error, "token id out of range");
      pos[i] = start + static_cast<int>(i);
    }
    Engine& e = engine;
    if (!e.Upload(e.ids, host.data(), host.size() * 4, error) ||
        !Check(hipMemcpy(static_cast<int*>(positions.get()) + first, pos.data(),
                         pos.size() * 4, hipMemcpyHostToDevice),
               "upload", error))
      return false;
    LaunchGather(embed.bf16(), static_cast<const int*>(e.ids.get()),
                 static_cast<int>(ids.size()), lm.shape.width,
                 hidden.f32() + std::size_t(first) * lm.shape.width, e.stream);
    return true;
  }
};

CodeLm::CodeLm(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CodeLm::~CodeLm() = default;
std::size_t CodeLm::weight_bytes() const {
  return impl_->weight_bytes;
}
const std::vector<float>& CodeLm::first_logits() const {
  return impl_->first_logits;
}
double CodeLm::last_codes_per_second() const {
  return impl_->codes_per_second;
}

std::unique_ptr<CodeLm> CodeLm::Load(const std::filesystem::path& safetensors,
                                     std::string* error) {
  auto impl = std::make_unique<Impl>();
  if (!impl->engine.Init(error))
    return nullptr;
  SafetensorsFile file;
  if (!file.Open(safetensors, impl->engine.stream, error))
    return nullptr;
  // Qwen3 geometry from the tensors: width from the embedding, heads from
  // the projections, FFN from gate_proj, layers by counting.
  const auto* embed = file.Find("model.embed_tokens.weight");
  const auto* q = file.Find("model.layers.0.self_attn.q_proj.weight");
  const auto* k = file.Find("model.layers.0.self_attn.k_proj.weight");
  const auto* gate = file.Find("model.layers.0.mlp.gate_proj.weight");
  const auto* q_norm = file.Find("model.layers.0.self_attn.q_norm.weight");
  if (embed == nullptr || embed->shape.size() != 2 || q == nullptr ||
      q->shape.size() != 2 || k == nullptr || k->shape.size() != 2 ||
      gate == nullptr || gate->shape.size() != 2 || q_norm == nullptr ||
      q_norm->count() != std::size_t(kHeadDim) ||
      embed->shape[0] < std::size_t(kFirstCode) + kCodes ||
      q->shape[1] != embed->shape[1] || q->shape[0] % kHeadDim != 0 ||
      k->shape[0] % kHeadDim != 0) {
    Fail(error, "not an ACE-Step 5 Hz LM (Qwen3 with audio codes): " +
                    safetensors.string());
    return nullptr;
  }
  LmShape& lm = impl->lm;
  lm.shape.width = static_cast<int>(embed->shape[1]);
  lm.shape.heads = static_cast<int>(q->shape[0] / kHeadDim);
  lm.shape.kv_heads = static_cast<int>(k->shape[0] / kHeadDim);
  lm.shape.ffn = static_cast<int>(gate->shape[0]);
  lm.layers = 0;
  while (file.Has("model.layers." + std::to_string(lm.layers) +
                  ".input_layernorm.weight"))
    ++lm.layers;
  if (lm.layers == 0 || lm.shape.heads % lm.shape.kv_heads != 0) {
    Fail(error, "unexpected ACE-Step LM geometry: " + safetensors.string());
    return nullptr;
  }
  impl->vocabulary = embed->shape[0];
  if (!file.Load("model.embed_tokens.weight", embed->count(), impl->embed, true,
                 error) ||
      !LoadEncoderLayers(file, "model.", lm.layers, lm.shape, impl->layers,
                         error) ||
      !file.Load("model.norm.weight", lm.shape.width, impl->norm, true,
                 error))
    return nullptr;
  if (const auto* head = file.Find("lm_head.weight");
      head != nullptr && head->shape == embed->shape &&
      !file.Load("lm_head.weight", head->count(), impl->head, true, error))
    return nullptr;
  if (!file.Finish(error))
    return nullptr;
  impl->weight_bytes = file.uploaded_bytes();
  return std::unique_ptr<CodeLm>(new CodeLm(std::move(impl)));
}

bool CodeLm::Generate(std::span<const std::uint32_t> conditional,
                      std::span<const std::uint32_t> unconditional, int count,
                      const CodeSampling& sampling, std::vector<int>* codes,
                      const std::function<bool(int)>& progress,
                      std::string* error) {
  Impl& m = *impl_;
  Engine& e = m.engine;
  const int w = m.lm.shape.width;
  const int layer_count = m.lm.layers;
  const bool guided = sampling.cfg != 1.0F;
  const std::span<const std::uint32_t> prompts[2] = {conditional,
                                                     unconditional};
  const int sequences = guided ? 2 : 1;
  int longest = 2;
  for (int q = 0; q < sequences; ++q) {
    if (prompts[q].empty())
      return Fail(error, "empty LM prompt");
    longest = std::max(longest, static_cast<int>(prompts[q].size()));
  }
  if (count <= 0)
    return Fail(error, "no audio codes requested");
  const int capacity = longest + count + 1;
  if (!m.hidden.Reserve(std::size_t(longest) * w * 4, error) ||
      !m.positions.Reserve(std::size_t(longest) * 4, error) ||
      !e.Reserve(longest, m.lm.shape, error))
    return false;
  for (int q = 0; q < sequences; ++q)
    if (!m.Allocate(m.cache[q], capacity, error))
      return false;
  // Prefill each prompt; its last hidden row moves to row q.
  std::vector<float> last(std::size_t(sequences) * w);
  for (int q = 0; q < sequences; ++q) {
    const int rows = static_cast<int>(prompts[q].size());
    if (!m.Embed(prompts[q], 0, 0, error))
      return false;
    for (int i = 0; i < layer_count; ++i)
      if (!m.Layer(i, q, 0, rows, true, error))
        return false;
    m.cache[q].length = rows;
    if (!e.Download(m.hidden.f32() + std::size_t(rows - 1) * w,
                    std::size_t(w) * 4, last.data() + std::size_t(q) * w,
                    error))
      return false;
  }
  if (!Check(hipMemcpy(m.hidden.get(), last.data(), last.size() * 4,
                       hipMemcpyHostToDevice),
             "upload", error))
    return false;
  std::mt19937_64 rng(sampling.seed);
  std::vector<int> order(kCodes);
  std::vector<double> weights(kCodes);
  std::vector<float> logits, scores(kCodes);
  codes->clear();
  const auto decode_start = std::chrono::steady_clock::now();
  for (int step = 0; step < count; ++step) {
    if (!m.Logits(sequences, &logits, error))
      return false;
    if (step == 0)
      m.first_logits = logits;
    // Classifier-free guidance, temperature, then nucleus (top-p) sampling.
    for (int k = 0; k < kCodes; ++k) {
      float value = logits[k];
      if (guided)
        value = logits[kCodes + k] +
                sampling.cfg * (logits[k] - logits[kCodes + k]);
      scores[k] =
          sampling.temperature > 0 ? value / sampling.temperature : value;
    }
    int token = 0;
    if (sampling.temperature <= 0) {
      token = static_cast<int>(std::max_element(scores.begin(), scores.end()) -
                               scores.begin());
    } else {
      for (int k = 0; k < kCodes; ++k)
        order[k] = k;
      std::sort(order.begin(), order.end(),
                [&](int a, int b) { return scores[a] > scores[b]; });
      const double top = scores[order[0]];
      double total = 0;
      for (int k = 0; k < kCodes; ++k) {
        weights[k] = std::exp(double(scores[order[k]]) - top);
        total += weights[k];
      }
      // Smallest prefix whose probability mass reaches top_p (>= 1 code).
      double mass = 0, kept = 0;
      int keep = 0;
      while (keep < kCodes) {
        mass += weights[keep] / total;
        kept += weights[keep];
        ++keep;
        if (mass >= sampling.top_p)
          break;
      }
      double draw = std::uniform_real_distribution<double>(0.0, kept)(rng);
      token = order[keep - 1];
      for (int k = 0; k < keep; ++k) {
        draw -= weights[k];
        if (draw <= 0) {
          token = order[k];
          break;
        }
      }
    }
    codes->push_back(token);
    if (progress && !progress(step + 1))
      return Fail(error, "cancelled");
    if (step + 1 == count)
      break;
    // Feed the code to every sequence (hidden row q).
    const std::uint32_t id = kFirstCode + static_cast<std::uint32_t>(token);
    for (int q = 0; q < sequences; ++q)
      if (!m.Embed(std::span<const std::uint32_t>(&id, 1), q, m.cache[q].length,
                   error))
        return false;
    for (int i = 0; i < layer_count; ++i)
      for (int q = 0; q < sequences; ++q)
        if (!m.Layer(i, q, q, 1, false, error))
          return false;
    for (int q = 0; q < sequences; ++q)
      ++m.cache[q].length;
  }
  const double seconds = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - decode_start)
                             .count();
  m.codes_per_second = seconds > 0 ? count / seconds : 0;
  return true;
}

std::vector<float> TurboTimesteps(int steps, float shift) {
  std::vector<float> t = MusicTimesteps(steps, shift);
  t.pop_back();
  return t;
}

}  // namespace gufo::models::ace_step
