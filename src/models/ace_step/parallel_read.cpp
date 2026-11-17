// Copy of src/models/wan22/parallel_read (one per user until the platform
// layer has it again).
// Written with AI assistance (Claude Code).
#include "src/models/ace_step/parallel_read.hpp"

#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

namespace gufo::models::ace_step {

std::size_t ReadFileParallel(platform::PlatformFile file, void* output,
                             std::size_t bytes, std::uint64_t offset,
                             unsigned readers) noexcept {
  constexpr std::size_t kSector = 4096;
  constexpr std::size_t kPiece = 8U << 20U;
  auto* target = static_cast<std::byte*>(output);
  const auto read_range = [](platform::PlatformFile handle,
                             std::byte* destination, std::size_t count,
                             std::uint64_t position) {
    std::size_t done = 0;
    while (done < count) {
      const std::size_t result = platform::ReadFileAt(
          handle, destination + done, count - done, position + done);
      if (result == 0) {
        break;
      }
      done += result;
    }
    return done;
  };
  const bool aligned = offset % kSector == 0 &&
                       reinterpret_cast<std::uintptr_t>(output) % kSector == 0;
  const std::size_t direct_bytes = aligned ? bytes / kSector * kSector : 0;
  std::atomic<std::size_t> next{0};
  // Lowest offset that could not be read completely (end of file or error).
  std::atomic<std::size_t> short_at{direct_bytes};
  const auto reader = [&]() {
    const platform::PlatformFile direct = platform::ReopenUnbuffered(file);
    const platform::PlatformFile handle =
        platform::FileValid(direct) ? direct : file;
    while (true) {
      const std::size_t piece = next.fetch_add(kPiece);
      if (piece >= direct_bytes || piece >= short_at.load()) {
        break;
      }
      const std::size_t count = std::min(kPiece, direct_bytes - piece);
      std::size_t done =
          read_range(handle, target + piece, count, offset + piece);
      if (done != count && handle != file) {
        done = read_range(file, target + piece, count, offset + piece);
      }
      if (done != count) {
        std::size_t current = short_at.load();
        while (piece + done < current &&
               !short_at.compare_exchange_weak(current, piece + done)) {
        }
      }
    }
    if (platform::FileValid(direct)) {
      platform::CloseFile(direct);
    }
  };
  if (direct_bytes != 0) {
    std::vector<std::thread> threads;
    try {
      for (unsigned index = 1; index < std::max(readers, 1U); ++index) {
        threads.emplace_back(reader);
      }
    } catch (...) {
      // Fewer threads only lower the read concurrency.
    }
    reader();
    for (auto& thread : threads) {
      thread.join();
    }
  }
  const std::size_t complete = short_at.load();
  if (complete != direct_bytes) {
    return complete;
  }
  return direct_bytes + read_range(file, target + direct_bytes,
                                   bytes - direct_bytes, offset + direct_bytes);
}

}  // namespace gufo::models::ace_step
