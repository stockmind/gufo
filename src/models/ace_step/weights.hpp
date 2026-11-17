#ifndef GUFO_MODELS_ACE_STEP_WEIGHTS_HPP_
#define GUFO_MODELS_ACE_STEP_WEIGHTS_HPP_

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include <gufo/ext/platform/mapped_file.hpp>
#include "src/models/ace_step/parallel_read.hpp"

namespace gufo::models::ace_step {

/// Sets `error` (when not null) and returns false.
bool Fail(std::string* error, std::string message);
/// Turns a HIP status into `error`; true on success.
bool Check(hipError_t status, const char* operation, std::string* error);

/// Owning device allocation.
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
  ~DeviceBuffer() { Release(); }
  bool Allocate(std::size_t bytes, std::string* error);
  /// Allocates only when the buffer is smaller than `bytes`.
  bool Reserve(std::size_t bytes, std::string* error);
  void Release();
  [[nodiscard]] std::uint16_t* bf16() const {
    return static_cast<std::uint16_t*>(data_);
  }
  [[nodiscard]] float* f32() const { return static_cast<float*>(data_); }
  [[nodiscard]] void* get() const { return data_; }
  [[nodiscard]] std::size_t bytes() const { return bytes_; }

private:
  void* data_{nullptr};
  std::size_t bytes_{0};
};

/// One safetensors file (F32 / BF16 tensors) read through pinned staging
/// buffers into device memory.
class SafetensorsFile {
public:
  struct Tensor {
    std::uint64_t offset{0};
    std::uint64_t bytes{0};
    bool f32{false};
    std::vector<std::size_t> shape;
    [[nodiscard]] std::size_t count() const;
  };

  SafetensorsFile() = default;
  SafetensorsFile(const SafetensorsFile&) = delete;
  SafetensorsFile& operator=(const SafetensorsFile&) = delete;
  ~SafetensorsFile();

  bool Open(const std::filesystem::path& path, hipStream_t stream,
            std::string* error);
  [[nodiscard]] const Tensor* Find(std::string_view name) const;
  [[nodiscard]] bool Has(std::string_view name) const {
    return Find(name) != nullptr;
  }
  /// Uploads the named tensor (checked against `count` elements) as BF16
  /// (`bf16`) or FP32 into `target`; F32 sources are converted on the device.
  bool Upload(std::string_view name, std::size_t count, void* target, bool bf16,
              std::string* error);
  /// Allocates `buffer` and uploads the tensor into it.
  bool Load(std::string_view name, std::size_t count, DeviceBuffer& buffer,
            bool bf16, std::string* error);
  /// Reads the named tensor to host FP32.
  bool ReadHost(std::string_view name, std::size_t count,
                std::vector<float>* out, std::string* error);
  /// Waits until the uploads issued so far have completed.
  bool Finish(std::string* error);
  [[nodiscard]] std::size_t uploaded_bytes() const { return uploaded_; }
  /// String members of the header's `__metadata__` (empty when absent).
  [[nodiscard]] const std::map<std::string, std::string>& metadata() const {
    return metadata_;
  }

private:
  bool Read(std::uint64_t offset, std::size_t bytes, int slot,
            const std::byte** data);

  std::filesystem::path path_;
  platform::PlatformFile file_{};
  std::map<std::string, Tensor, std::less<>> tensors_;
  std::map<std::string, std::string> metadata_;
  hipStream_t stream_{nullptr};
  void* staging_[2]{};
  DeviceBuffer staging_device_[2];
  hipEvent_t staged_[2]{};
  int next_{0};
  std::size_t uploaded_{0};
};

/// FP32 <-> BF16 on the host (round to nearest even).
std::uint16_t HostBf16(float value);
float HostFloat(std::uint16_t value);

}  // namespace gufo::models::ace_step

#endif  // GUFO_MODELS_ACE_STEP_WEIGHTS_HPP_
