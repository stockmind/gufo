#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstring>
#include <utility>

#include <gufo/ext/json.hpp>
#include "src/models/ace_step/weights.hpp"

namespace gufo::models::ace_step {
namespace {

constexpr std::size_t kStagingBytes = 32U << 20U;
constexpr std::size_t kSector = 4096;
constexpr std::size_t kStagingSlack = 2U * kSector;

__global__ void ConvertKernel(const float* input, std::size_t count,
                              std::uint16_t* output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  const unsigned bits = __float_as_uint(input[index]);
  output[index] = (bits & 0x7FFFFFFFU) > 0x7F800000U
                      ? static_cast<std::uint16_t>((bits >> 16U) | 0x40U)
                      : static_cast<std::uint16_t>(
                            (bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U);
}

__global__ void WidenKernel(const std::uint16_t* input, std::size_t count,
                            float* output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    output[index] = __uint_as_float(static_cast<unsigned>(input[index]) << 16U);
}

}  // namespace

bool Fail(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
  return false;
}

bool Check(hipError_t status, const char* operation, std::string* error) {
  if (status == hipSuccess)
    return true;
  return Fail(error, std::string("ACE-Step ") + operation + ": " +
                         hipGetErrorString(status));
}

std::uint16_t HostBf16(float value) {
  unsigned bits = 0;
  std::memcpy(&bits, &value, 4);
  if ((bits & 0x7FFFFFFFU) > 0x7F800000U)
    return static_cast<std::uint16_t>((bits >> 16U) | 0x40U);
  return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16U) & 1U)) >>
                                    16U);
}

float HostFloat(std::uint16_t value) {
  const unsigned bits = static_cast<unsigned>(value) << 16U;
  float out = 0;
  std::memcpy(&out, &bits, 4);
  return out;
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      bytes_(std::exchange(other.bytes_, 0)) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    Release();
    data_ = std::exchange(other.data_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
  }
  return *this;
}

bool DeviceBuffer::Allocate(std::size_t bytes, std::string* error) {
  Release();
  bytes = std::max<std::size_t>(bytes, 256);
  if (!Check(hipMalloc(&data_, bytes), "device allocation", error)) {
    data_ = nullptr;
    return false;
  }
  bytes_ = bytes;
  return true;
}

bool DeviceBuffer::Reserve(std::size_t bytes, std::string* error) {
  return bytes <= bytes_ || Allocate(bytes, error);
}

void DeviceBuffer::Release() {
  if (data_ != nullptr)
    (void)hipFree(data_);
  data_ = nullptr;
  bytes_ = 0;
}

std::size_t SafetensorsFile::Tensor::count() const {
  std::size_t n = 1;
  for (std::size_t d : shape)
    n *= d;
  return n;
}

SafetensorsFile::~SafetensorsFile() {
  if (stream_ != nullptr)
    (void)hipStreamSynchronize(stream_);
  if (platform::FileValid(file_))
    platform::CloseFile(file_);
  for (void* buffer : staging_)
    if (buffer != nullptr)
      (void)hipHostFree(buffer);
  for (hipEvent_t event : staged_)
    if (event != nullptr)
      (void)hipEventDestroy(event);
}

bool SafetensorsFile::Open(const std::filesystem::path& path,
                           hipStream_t stream, std::string* error) {
  path_ = path;
  stream_ = stream;
  std::string open_error;
  file_ = platform::OpenReadFile(path, &open_error);
  if (!platform::FileValid(file_))
    return Fail(error, "cannot open " + path.string() + ": " + open_error);
  const auto file_size = platform::FileSizeBytes(file_);
  std::uint64_t header_size = 0;
  if (!file_size || platform::ReadFileAt(file_, &header_size, 8, 0) != 8 ||
      header_size == 0 || header_size > (64U << 20U) ||
      header_size > *file_size - 8)
    return Fail(error, "invalid safetensors header in " + path.string());
  std::string text(static_cast<std::size_t>(header_size), '\0');
  if (platform::ReadFileAt(file_, text.data(), text.size(), 8) != text.size())
    return Fail(error, "short read of " + path.string());
  json::Value header;
  try {
    header = json::parse(text);
  } catch (const std::exception& e) {
    return Fail(error, path.string() + ": " + e.what());
  }
  const std::uint64_t payload = 8 + header_size;
  metadata_.clear();
  for (const auto& [name, entry] : header.members()) {
    if (name == "__metadata__") {
      if (entry.is_object())
        for (const auto& [key, value] : entry.members())
          if (value.is_string())
            metadata_[key] = value.str();
      continue;
    }
    const json::Value* shape = entry.find("shape");
    const json::Value* offsets = entry.find("data_offsets");
    const std::string dtype = entry.member_str("dtype");
    if ((dtype != "F32" && dtype != "BF16") || shape == nullptr ||
        !shape->is_array() || offsets == nullptr || !offsets->is_array() ||
        offsets->size() != 2)
      return Fail(error, "unsupported tensor " + name + " in " + path.string());
    Tensor tensor;
    tensor.f32 = dtype == "F32";
    std::uint64_t bytes = tensor.f32 ? 4 : 2;
    for (const auto& dim : shape->items()) {
      tensor.shape.push_back(dim.as_size());
      bytes *= tensor.shape.back();
    }
    const std::uint64_t begin = offsets->items()[0].as_size();
    const std::uint64_t end = offsets->items()[1].as_size();
    if (end < begin || end - begin != bytes || payload + end > *file_size)
      return Fail(error, "invalid tensor extent " + name);
    tensor.offset = payload + begin;
    tensor.bytes = bytes;
    tensors_[name] = std::move(tensor);
  }
  for (int slot = 0; slot < 2; ++slot) {
    if (!Check(hipHostMalloc(&staging_[slot], kStagingBytes + kStagingSlack),
               "staging allocation", error) ||
        !staging_device_[slot].Allocate(kStagingBytes, error) ||
        !Check(hipEventCreate(&staged_[slot]), "staging event", error))
      return false;
    if (!Check(hipEventRecord(staged_[slot], stream_), "staging event", error))
      return false;
  }
  return true;
}

const SafetensorsFile::Tensor* SafetensorsFile::Find(
    std::string_view name) const {
  const auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

bool SafetensorsFile::Read(std::uint64_t offset, std::size_t bytes, int slot,
                           const std::byte** data) {
  const std::uint64_t aligned = offset / kSector * kSector;
  const auto lead = static_cast<std::size_t>(offset - aligned);
  const std::size_t span = (lead + bytes + kSector - 1U) / kSector * kSector;
  auto* buffer = static_cast<std::byte*>(staging_[slot]);
  if (ReadFileParallel(file_, buffer, span, aligned) < lead + bytes)
    return false;
  *data = buffer + lead;
  return true;
}

bool SafetensorsFile::Upload(std::string_view name, std::size_t count,
                             void* target, bool bf16, std::string* error) {
  const Tensor* tensor = Find(name);
  if (tensor == nullptr)
    return Fail(error, "missing tensor " + std::string(name) + " in " +
                           path_.filename().string());
  if (tensor->count() != count)
    return Fail(error, "unexpected size of tensor " + std::string(name));
  // FP32 targets from BF16 sources are widened through the device staging.
  const std::size_t source_element = tensor->f32 ? 4 : 2;
  const std::size_t target_element = bf16 ? 2 : 4;
  const std::size_t per_chunk = kStagingBytes / 4;  // elements
  for (std::size_t first = 0; first < count; first += per_chunk) {
    const std::size_t n = std::min(per_chunk, count - first);
    const int slot = next_;
    next_ ^= 1;
    if (!Check(hipEventSynchronize(staged_[slot]), "weight staging", error))
      return false;
    const std::byte* data = nullptr;
    if (!Read(tensor->offset + first * source_element, n * source_element, slot,
              &data))
      return Fail(error, "short read of tensor " + std::string(name));
    auto* out = static_cast<std::byte*>(target) + first * target_element;
    if (tensor->f32 == !bf16) {
      if (!Check(hipMemcpyAsync(out, data, n * source_element,
                                hipMemcpyHostToDevice, stream_),
                 "weight upload", error))
        return false;
    } else {
      if (!Check(hipMemcpyAsync(staging_device_[slot].get(), data,
                                n * source_element, hipMemcpyHostToDevice,
                                stream_),
                 "weight upload", error))
        return false;
      const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
      if (tensor->f32)
        ConvertKernel<<<blocks, 256, 0, stream_>>>(
            staging_device_[slot].f32(), n,
            reinterpret_cast<std::uint16_t*>(out));
      else
        WidenKernel<<<blocks, 256, 0, stream_>>>(
            staging_device_[slot].bf16(), n, reinterpret_cast<float*>(out));
    }
    if (!Check(hipEventRecord(staged_[slot], stream_), "weight upload", error))
      return false;
  }
  uploaded_ += count * target_element;
  return true;
}

bool SafetensorsFile::Load(std::string_view name, std::size_t count,
                           DeviceBuffer& buffer, bool bf16,
                           std::string* error) {
  return buffer.Allocate(count * (bf16 ? 2 : 4), error) &&
         Upload(name, count, buffer.get(), bf16, error);
}

bool SafetensorsFile::ReadHost(std::string_view name, std::size_t count,
                               std::vector<float>* out, std::string* error) {
  const Tensor* tensor = Find(name);
  if (tensor == nullptr || tensor->count() != count)
    return Fail(error, "missing or mis-sized tensor " + std::string(name));
  std::vector<std::byte> raw(tensor->bytes);
  if (platform::ReadFileAt(file_, raw.data(), raw.size(), tensor->offset) !=
      raw.size())
    return Fail(error, "short read of tensor " + std::string(name));
  out->resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    if (tensor->f32) {
      std::memcpy(&(*out)[i], raw.data() + i * 4, 4);
    } else {
      std::uint16_t v = 0;
      std::memcpy(&v, raw.data() + i * 2, 2);
      (*out)[i] = HostFloat(v);
    }
  }
  return true;
}

bool SafetensorsFile::Finish(std::string* error) {
  return Check(hipStreamSynchronize(stream_), "weight upload", error);
}

}  // namespace gufo::models::ace_step
