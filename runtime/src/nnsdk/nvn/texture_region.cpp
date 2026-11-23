#include "nnsdk/nvn/texture_region.h"
#include "nnsdk/nvn/texture_view.h"
#include "nnsdk/nvn/resource_access.h"

namespace nvn {
namespace {
VkExtent3D image_extent(const Texture& t, uint32_t level) {
    auto extent = mip_extent(t, level);
    if (t.desc.target == kTarget1D || t.desc.target == kTarget1DArray) extent.height = 1;
    const auto depth = t.desc.target == kTarget3D ? std::max(uint32_t(t.desc.depth) >> std::min(level, 31u), 1u) : 1u;
    return {extent.width, extent.height, depth};
}

bool image_region(const Texture& t, const TextureViewDesc& view, const CopyRegion& input, TextureRegion& out) {
    if (!resolve_texture_region(t.desc.target, view, input, std::max(t.desc.levels, 1), t.layers, out)) return false;
    const auto extent = image_extent(t, out.level);
    const auto& box = out.box;
    return int64_t(box.x) + box.width <= extent.width && int64_t(box.y) + box.height <= extent.height &&
           int64_t(box.z) + box.depth <= extent.depth;
}
} // namespace

bool plan_texture_copy(const cmd::CopyTexture& c, TextureCopyPlan& plan) {
    const auto& src = *c.src;
    const auto& dst = *c.dst;
    if (!image_region(src, c.src_view, c.src_region, plan.source) ||
        !image_region(dst, c.dst_view, c.dst_region, plan.destination)) {
        NVN_LOG("CopyTexture: region outside image or view\n");
        return false;
    }
    const auto& a = plan.source.box;
    const auto& b = plan.destination.box;
    const bool src_3d = src.desc.target == kTarget3D, dst_3d = dst.desc.target == kTarget3D;
    const bool matching_slices =
        (src_3d ? uint32_t(a.depth) : plan.source.layers) == (dst_3d ? uint32_t(b.depth) : plan.destination.layers);
    plan.aspects = src.aspect;
    plan.direct = src.vk_format == dst.vk_format && a.width == b.width && a.height == b.height && matching_slices;
    if (src.aspect != dst.aspect || std::max(src.desc.samples, 1) != std::max(dst.desc.samples, 1)) {
        NVN_LOG("CopyTexture: incompatible image aspects or sample counts\n");
        return false;
    }
    plan.filter = VK_FILTER_NEAREST;
    if (!plan.direct) {
        // These are the blit layer restrictions of the Vulkan 1.1 backend.
        if (plan.source.layers != plan.destination.layers || src.desc.samples > 1 ||
            ((src_3d || dst_3d) && (plan.source.layer || plan.destination.layer || plan.source.layers != 1)) ||
            (src.aspect != VK_IMAGE_ASPECT_COLOR_BIT && src.vk_format != dst.vk_format)) {
            NVN_LOG("CopyTexture: incompatible blit layers, samples or depth/stencil formats\n");
            return false;
        }
        VkFormatProperties from, to;
        vkGetPhysicalDeviceFormatProperties(vk()->gpu, src.vk_format, &from);
        vkGetPhysicalDeviceFormatProperties(vk()->gpu, dst.vk_format, &to);
        if (!(from.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) ||
            !(to.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
            NVN_LOG("CopyTexture: blit unavailable for Vulkan formats %d -> %d\n", src.vk_format, dst.vk_format);
            return false;
        }
        if (src.aspect == VK_IMAGE_ASPECT_COLOR_BIT &&
            (from.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
            plan.filter = VK_FILTER_LINEAR;
    }
    const auto extent = image_extent(dst, 0);
    plan.replaces_destination = dst.desc.levels == 1 && plan.destination.layers == dst.layers && !b.x && !b.y && !b.z &&
                                uint32_t(b.width) == extent.width && uint32_t(b.height) == extent.height &&
                                uint32_t(b.depth) == extent.depth;
    return true;
}

void TextureCopyPlan::record(VkCommandBuffer cmd, VkImage src, VkImage dst) const {
    const auto sr = source.range(aspects), dr = destination.range(aspects);
    const VkImageMemoryBarrier before[]{image_dependency(src, sr, prior_writes.access, transfer_read.access),
                                        image_dependency(dst, dr, prior_writes.access, transfer_write.access)};
    vkCmdPipelineBarrier(cmd, prior_writes.stage, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, before);
    const VkImageSubresourceLayers sl{aspects, source.level, source.layer, source.layers};
    const VkImageSubresourceLayers dl{aspects, destination.level, destination.layer, destination.layers};
    const auto& a = source.box;
    const auto& b = destination.box;
    const VkOffset3D from{a.x, a.y, a.z}, to{b.x, b.y, b.z};
    if (direct) {
        VkImageCopy r{sl, from, dl, to, {uint32_t(a.width), uint32_t(a.height), uint32_t(std::max(a.depth, b.depth))}};
        vkCmdCopyImage(cmd, src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    } else {
        VkImageBlit r{sl,
                      {from, {a.x + a.width, a.y + a.height, a.z + a.depth}},
                      dl,
                      {to, {b.x + b.width, b.y + b.height, b.z + b.depth}}};
        vkCmdBlitImage(cmd, src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL, 1, &r, filter);
    }
    sync_image(cmd, dst, dr, transfer_write, later_access);
}
} // namespace nvn
