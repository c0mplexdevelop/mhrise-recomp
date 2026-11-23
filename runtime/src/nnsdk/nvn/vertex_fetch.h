#pragma once
#include <cstdint>
#include <vector>

namespace nvn {
// Four bits per generic, using the translator's Options::input_types convention.
enum class VertexFetch : uint8_t { Float, Signed, Unsigned, SignedScaled, UnsignedScaled };
// A float register in the base translation carries the guest register's bits. A 32-bit
// scaled fetch is the exception: Vulkan has no scaled format for it, so convert numerically.
// Empty output on success means the existing interface already matches.
bool specialize_vertex_fetch(const std::vector<uint32_t>& code, uint64_t types, std::vector<uint32_t>& output);
} // namespace nvn
