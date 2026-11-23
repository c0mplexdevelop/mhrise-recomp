#include "nnsdk/nvn/texture_view.h"
#include "nnsdk/nvn/bcn.h"
#include "nnsdk/nvn/formats.h"
#include <algorithm>
#include <cstring>
#include <iterator>

namespace nvn {

void MipResidency::publish(uint32_t count, std::vector<uint8_t> next) {
    if (std::none_of(next.begin(), next.end(), [](uint8_t issue) { return issue != 0; })) next.clear();
    if (levels == count && issues == next) return;
    levels = count;
    issues = std::move(next);
    ++revision;
}

void MipResidency::encoded(uint32_t layer, uint32_t mip) {
    const size_t slot = size_t{layer} * levels + mip;
    if (slot < issues.size() && (issues[slot] & Encoding)) {
        issues[slot] &= ~Encoding;
        ++revision;
        if (std::none_of(issues.begin(), issues.end(), [](uint8_t issue) { return issue != 0; })) issues.clear();
    }
}

void MipResidency::restrict_sample(VkImageSubresourceRange& range) const {
    if (issues.empty()) return;
    auto ready = [&](uint32_t mip, uint8_t blocking) {
        for (uint32_t i = 0; i < range.layerCount; ++i) {
            const size_t slot = size_t{range.baseArrayLayer + i} * levels + mip;
            if (mip >= levels || slot >= issues.size() || (issues[slot] & blocking)) return false;
        }
        return true;
    };
    const auto first = range.baseMipLevel;
    // Prefer the complete tail. When a bounded view only contains partially backed
    // levels, their uploaded pages still contain real texels and zero-filled holes.
    // Keep their format, dimensions and channel mapping.
    for (uint8_t blocking : {uint8_t(Unmapped | InvalidAstc | Encoding), uint8_t(InvalidAstc | Encoding)}) {
        auto end = first + range.levelCount;
        while (end > first && !ready(end - 1, blocking)) --end;
        if (end == first) continue;
        auto begin = end - 1;
        while (begin > first && ready(begin - 1, blocking)) --begin;
        range.baseMipLevel = begin;
        range.levelCount = end - begin;
        return;
    }
    range.levelCount = 0;
}

bool is_depth_format(VkFormat f) {
    return f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_X8_D24_UNORM_PACK32 || f == VK_FORMAT_D32_SFLOAT ||
           f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_S8_UINT;
}

VkFormat storage_format(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_UNORM;
        case VK_FORMAT_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_UNORM;
        case VK_FORMAT_A8B8G8R8_SRGB_PACK32: return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
        default: return f;
    }
}

namespace {
bool cube(int target) { return target == kTargetCube || target == kTargetCubeArray; }
int dimensions(int target) {
    switch (target) {
        case kTarget1D: case kTarget1DArray: return 1;
        case kTarget2D: case kTarget2DArray: case kTargetRectangle: case kTarget2DMultisample:
        case kTarget2DMultisampleArray: case kTargetCube: case kTargetCubeArray: return 2;
        case kTarget3D: return 3;
        default: return 0;
    }
}
VkImageViewType view_type(int target, ViewUse use) {
    if (use == ViewUse::Attachment && dimensions(target) == 2) return VK_IMAGE_VIEW_TYPE_2D;
    switch (target) {
        case kTarget1D: return VK_IMAGE_VIEW_TYPE_1D;
        case kTarget1DArray: return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
        case kTarget3D: return VK_IMAGE_VIEW_TYPE_3D;
        case kTargetCube: return VK_IMAGE_VIEW_TYPE_CUBE;
        case kTargetCubeArray: return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
        case kTarget2DArray: case kTarget2DMultisampleArray: return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        default: return VK_IMAGE_VIEW_TYPE_2D;
    }
}
VkFormat view_format(const Texture& t, const TextureViewDesc& v) {
    if (!t.mutable_format || t.astc || !(v.flags & kViewFormat)) return t.vk_format;
    const auto* info = format_info(v.format);
    if (!info || info->vk == VK_FORMAT_UNDEFINED) return t.vk_format;
    if (is_depth_format(t.vk_format) && is_depth_format(info->vk)) return t.vk_format;
    if (!t.bcn) return info->vk;
    const auto plain = bcn::decoded_format(info->vk);
    return plain != VK_FORMAT_UNDEFINED && bcn::texel_bytes(bcn::kind_of(info->vk)) == bcn::texel_bytes(static_cast<bcn::Kind>(t.bcn))
           ? plain : t.vk_format;
}
VkComponentMapping sampled_components(const TextureDesc& d, const TextureViewDesc& v) {
    const bool defaults = !(d.swizzle[0] | d.swizzle[1] | d.swizzle[2] | d.swizzle[3]);
    VkComponentSwizzle channels[4];
    constexpr VkComponentSwizzle map[]{VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ONE, VK_COMPONENT_SWIZZLE_R,
                                       VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
    for (unsigned i = 0; i < 4; ++i) {
        unsigned c = (v.flags & kViewSwizzle) ? v.swizzle[i] : i + 2;
        if (!defaults && c >= 2 && c <= 5) c = d.swizzle[c-2];
        channels[i] = c < std::size(map) && c != i+2 ? map[c] : VK_COMPONENT_SWIZZLE_IDENTITY;
    }
    return {channels[0], channels[1], channels[2], channels[3]};
}
} // namespace

bool select_shader_target(const Texture& t, uint32_t reflected_type, TextureViewDesc& v) {
    constexpr int targets[]{kTarget1D, kTarget1DArray, kTarget2D, kTarget2DArray, kTarget3D,
                            kTargetCube, kTargetCubeArray, kTargetBuffer, kTargetRectangle};
    if (reflected_type >= std::size(targets)) return false;
    const int want = targets[reflected_type];
    if (!dimensions(want) || dimensions(want) != dimensions(t.desc.target) || (cube(want) && !cube(t.desc.target))) return false;
    v.flags |= kViewTarget;
    v.target = want;
    return true;
}

VkImageSubresourceRange view_range(const Texture& t, const TextureViewDesc& v, ViewUse use) {
    const unsigned total = std::max(t.desc.levels, 1);
    unsigned mip = (v.flags & kViewLevels) ? v.base_level : 0;
    const unsigned layer = (v.flags & kViewLayers) ? v.base_layer : 0;
    if (mip >= total || layer >= t.layers) return {};
    unsigned levels = (v.flags & kViewLevels) && v.levels > 0 ? std::min<unsigned>(v.levels, total-mip) : total-mip;
    unsigned layers = (v.flags & kViewLayers) && v.layers > 0 ? std::min<unsigned>(v.layers, t.layers-layer) : t.layers-layer;
    const int target = (v.flags & kViewTarget) ? v.target : t.desc.target;
    if (!dimensions(target) || dimensions(target) != dimensions(t.desc.target) ||
        (cube(target) && !cube(t.desc.target)) || (use == ViewUse::Attachment && target == kTarget3D)) return {};
    const auto type = view_type(target, use);
    if (use != ViewUse::Sampled) levels = 1;
    if (use == ViewUse::Attachment || type == VK_IMAGE_VIEW_TYPE_2D || type == VK_IMAGE_VIEW_TYPE_1D) layers = 1;
    if (type == VK_IMAGE_VIEW_TYPE_CUBE) {
        if (layers < 6) return {};
        layers = 6;
    }
    if (type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY && layers % 6) return {};
    auto aspect = t.aspect;
    if (use != ViewUse::Attachment && (aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) {
        const int mode = (v.flags & kViewDepthStencil) ? v.depth_stencil_mode : t.desc.depth_stencil_mode;
        aspect = mode == 1 && (aspect & VK_IMAGE_ASPECT_STENCIL_BIT) ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    VkImageSubresourceRange range{aspect, mip, levels, layer, layers};
    if (use == ViewUse::Sampled) t.residency.restrict_sample(range);
    return range;
}

ImageViewSpec describe_view(const Texture& t, const TextureViewDesc& v, ViewUse use) {
    ImageViewSpec spec{};
    const int target = (v.flags & kViewTarget) ? v.target : t.desc.target;
    spec.type = view_type(target, use);
    spec.range = view_range(t, v, use);
    spec.format = view_format(t, v);
    switch (use) {
        case ViewUse::Sampled:
            spec.components = sampled_components(t.desc, v);
            spec.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
            break;
        case ViewUse::Storage:
            spec.format = storage_format(spec.format);
            spec.usage = VK_IMAGE_USAGE_STORAGE_BIT;
            break;
        case ViewUse::Attachment:
            spec.usage = is_depth_format(spec.format) ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            break;
    }
    return spec;
}

VkExtent2D mip_extent(const Texture& t, uint32_t level) {
    const auto shift = std::min(level, 31u);
    return {std::max(1u, uint32_t(t.desc.width) >> shift), std::max(1u, uint32_t(t.desc.height) >> shift)};
}

bool same_attachment_view(const Texture& t, const TextureViewDesc& a, const TextureViewDesc& b) {
    if (std::memcmp(&a, &b, sizeof a) == 0) return true;
    if (!t.image) return false; // format substitution and layer count are decided at allocation
    const auto av = describe_view(t, a, ViewUse::Attachment);
    const auto bv = describe_view(t, b, ViewUse::Attachment);
    return av.range.aspectMask && bv.range.aspectMask && av.type == bv.type && av.format == bv.format &&
           av.range.baseMipLevel == bv.range.baseMipLevel && av.range.baseArrayLayer == bv.range.baseArrayLayer;
}

bool ranges_overlap(const VkImageSubresourceRange& a, const VkImageSubresourceRange& b) {
    return a.levelCount && b.levelCount && a.layerCount && b.layerCount &&
           (a.aspectMask & b.aspectMask) && a.baseMipLevel < uint64_t(b.baseMipLevel) + b.levelCount &&
           b.baseMipLevel < uint64_t(a.baseMipLevel) + a.levelCount &&
           a.baseArrayLayer < uint64_t(b.baseArrayLayer) + b.layerCount &&
           b.baseArrayLayer < uint64_t(a.baseArrayLayer) + a.layerCount;
}
} // namespace nvn
