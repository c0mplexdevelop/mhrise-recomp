#pragma once
#include <algorithm>
#include "nnsdk/nvn/commands.h"

namespace nvn {

// NVN puts array layers in Y for 1D arrays and in Z for 2D arrays and cubes.
// After decoding, only an actual volume has more than one spatial Z slice.
struct TextureRegion {
    uint32_t level, layer, layers;
    CopyRegion box;
    bool operator==(const TextureRegion&) const = default;
    VkImageSubresourceRange range(VkImageAspectFlags aspects) const { return {aspects, level, 1, layer, layers}; }
};

inline int transfer_level(const TextureViewDesc& view) { return view.flags & kViewLevels ? view.base_level : 0; }
inline int transfer_base_layer(const TextureViewDesc& view) { return view.flags & kViewLayers ? view.base_layer : 0; }

inline bool resolve_texture_region(int target, const TextureViewDesc& view, const CopyRegion& input, uint32_t levels,
                                   uint32_t layers, TextureRegion& out) {
    const int level = transfer_level(view);
    const int base = transfer_base_layer(view);
    if (level < 0 || uint32_t(level) >= levels || base < 0 || uint32_t(base) >= layers || input.x < 0 || input.y < 0 ||
        input.z < 0 || input.width <= 0 || input.height <= 0 || input.depth < 0)
        return false;
    auto box = input;
    if (box.depth == 0) box.depth = 1;
    uint32_t first = 0, count = 1;
    switch (target) {
    case kTarget1DArray:
        first = box.y;
        count = box.height;
        box.y = 0;
        box.height = 1;
        break;
    case kTarget2DArray:
    case kTarget2DMultisampleArray:
    case kTargetCube:
    case kTargetCubeArray:
        first = box.z;
        count = box.depth;
        box.z = 0;
        box.depth = 1;
        break;
    default:
        break;
    }
    uint32_t available = layers - uint32_t(base);
    if ((view.flags & kViewLayers) && view.layers > 0) available = std::min(available, uint32_t(view.layers));
    if (first >= available || count > available - first) return false;
    out = {uint32_t(level), uint32_t(base) + first, count, box};
    return true;
}

struct TextureCopyPlan {
    TextureRegion source, destination;
    VkImageAspectFlags aspects;
    bool direct, replaces_destination;
    VkFilter filter;
    void record(VkCommandBuffer cmd, VkImage src, VkImage dst) const;
};
bool plan_texture_copy(const cmd::CopyTexture& command, TextureCopyPlan& plan);
} // namespace nvn
