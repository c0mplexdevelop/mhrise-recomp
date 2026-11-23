#pragma once

#include "nnsdk/nvn/state.h"
#include "nvn_shader.h"

namespace nvn {

// Maxwell exports are indexed by MRT; NVN write enables have four channel bits per MRT.
inline uint32_t color_write_channels(uint32_t masks, uint32_t exports) {
    uint32_t channels = 0;
    for (unsigned i = 0; i < 8; ++i)
        if (exports & (1u << i)) channels |= masks & (15u << (4 * i));
    return channels;
}

struct DrawAttachmentAccess {
    uint32_t color_channels;
    VkImageAspectFlags ds_reads, ds_writes;
    VkImageAspectFlags writes(int slot) const {
        if (slot < 0) return ds_writes;
        return (color_channels & (15u << (4 * slot))) ? VK_IMAGE_ASPECT_COLOR_BIT : 0;
    }
};

inline DrawAttachmentAccess draw_attachment_access(uint32_t masks, uint32_t exports, const DepthStencilState& ds,
                                                  const uint32_t stencil_mask[2]) {
    DrawAttachmentAccess a{color_write_channels(masks, exports), 0, 0};
    if (ds.depth_test) {
        a.ds_reads |= VK_IMAGE_ASPECT_DEPTH_BIT;
        if (ds.depth_write) a.ds_writes |= VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    if (ds.stencil_test) {
        a.ds_reads |= VK_IMAGE_ASPECT_STENCIL_BIT;
        auto modifies = [](unsigned op) { return op >= 2 && op <= 8; }; // same operations as vk_stencil_op
        if (((stencil_mask[0] & 255) && (modifies(ds.front_fail) || modifies(ds.front_zfail) || modifies(ds.front_zpass))) ||
            ((stencil_mask[1] & 255) && (modifies(ds.back_fail) || modifies(ds.back_zfail) || modifies(ds.back_zpass))))
            a.ds_writes |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    return a;
}

inline bool shader_writes_storage(const nvnshader::Resource& r) {
    using R = nvnshader::Resource;
    // The existing translator reflects no write qualifier for a storage texel buffer.
    return r.kind == R::ImageBuffer || ((r.kind == R::StorageBuffer || r.kind == R::Image) && r.is_written);
}

struct AccessScope {
    VkPipelineStageFlags stage;
    VkAccessFlags access;
};
inline constexpr AccessScope prior_writes{VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT};
inline constexpr AccessScope later_access{VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
inline constexpr AccessScope transfer_read{VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT};
inline constexpr AccessScope transfer_write{VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT};

inline void sync_memory(VkCommandBuffer cmd, AccessScope from, AccessScope to) {
    VkMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b.srcAccessMask = from.access;
    b.dstAccessMask = to.access;
    vkCmdPipelineBarrier(cmd, from.stage, to.stage, 0, 1, &b, 0, nullptr, 0, nullptr);
}

inline VkImageMemoryBarrier image_dependency(VkImage image, VkImageSubresourceRange range, VkAccessFlags from,
                                            VkAccessFlags to, VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL) {
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = from;
    b.dstAccessMask = to;
    b.oldLayout = b.newLayout = layout;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = range;
    return b;
}

inline void sync_image(VkCommandBuffer cmd, VkImage image, VkImageSubresourceRange range,
                       AccessScope from, AccessScope to, VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL) {
    auto b = image_dependency(image, range, from.access, to.access, layout);
    vkCmdPipelineBarrier(cmd, from.stage, to.stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

}  // namespace nvn
