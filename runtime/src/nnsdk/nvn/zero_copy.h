#pragma once
// Memory pools the GPU reads in place (zero_copy.cpp).

#include <cstddef>
#include <cstdint>

#include "nnsdk/nvn/vk.h"

namespace nvn::zero_copy {

// nvnMemoryPoolInitialize: if the switch is on and the pool qualifies, its pages become memory the GPU can read
// directly (the game's data copied over first). true if it did.
bool adopt(void* storage, size_t size, int flags);
// nvnMemoryPoolFinalize: the pages go back to the heap with what they hold now. After the GPU is idle.
void release(void* storage);
// The executor writes game memory itself while it replays a submit (UpdateUniformBuffer, CopyBuffer, ClearBuffer):
// the draws recorded before that would read it later on the GPU, already changed. A pool written that way goes back
// to being copied from then on. true if any pool is shared (for the scan before each submit).
void cpu_writes(uint64_t address, uint64_t size);
bool any();
// [address, address + size) inside one shared pool that's read in place: its buffer and the offset there.
bool find(uint64_t address, uint64_t size, VkBuffer* buffer, VkDeviceSize* offset);

}  // namespace nvn::zero_copy
