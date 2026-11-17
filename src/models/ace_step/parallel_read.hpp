// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_ACE_STEP_PARALLEL_READ_HPP_
#define GUFO_MODELS_ACE_STEP_PARALLEL_READ_HPP_

#include <cstddef>
#include <cstdint>

#include <gufo/ext/platform/mapped_file.hpp>

namespace gufo::models::ace_step {

/// Reads `bytes` at `offset` with `readers` threads. When `offset` and
/// `output` are 4096-byte aligned, whole sectors go through unbuffered
/// handles in 8 MiB pieces (several requests in flight, no cache copy, no
/// standby pages); the unaligned tail and any failed piece use `file`.
/// Returns the bytes read: less than `bytes` only at end of file or on
/// failure. (The 1.x fork had this in core/platform; the 0.2 base does not,
/// so the ACE-Step loader keeps their own copy on the public platform calls.)
std::size_t ReadFileParallel(platform::PlatformFile file, void* output,
                             std::size_t bytes, std::uint64_t offset,
                             unsigned readers = 4) noexcept;

}  // namespace gufo::models::ace_step

#endif  // GUFO_MODELS_ACE_STEP_PARALLEL_READ_HPP_
