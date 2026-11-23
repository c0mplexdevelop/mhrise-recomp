#pragma once

#include "nnsdk/nvn/objects.h"

namespace nvn {

enum class ViewUse { Sampled, Storage, Attachment };

// The same description keys the VkImageView cache and supplies framebuffer compatibility.
struct ImageViewSpec {
    VkImageViewType type;
    VkFormat format;
    VkImageSubresourceRange range;
    VkComponentMapping components;
    VkImageUsageFlags usage;
};

bool is_depth_format(VkFormat format);
VkFormat storage_format(VkFormat format);
bool select_shader_target(const Texture& texture, uint32_t reflected_type, TextureViewDesc& view);
VkImageSubresourceRange view_range(const Texture& texture, const TextureViewDesc& view, ViewUse use);
ImageViewSpec describe_view(const Texture& texture, const TextureViewDesc& view, ViewUse use);
bool same_attachment_view(const Texture& texture, const TextureViewDesc& a, const TextureViewDesc& b);
VkExtent2D mip_extent(const Texture& texture, uint32_t level);
bool ranges_overlap(const VkImageSubresourceRange& a, const VkImageSubresourceRange& b);

} // namespace nvn
