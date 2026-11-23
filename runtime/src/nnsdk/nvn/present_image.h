#pragma once
#include "nnsdk/nvn/vk.h"

namespace nvn {
struct PresentImage {
    VkImage image = VK_NULL_HANDLE;
    // Reuse follows acquisition of this swapchain image, across all producers.
    VkSemaphore ready = VK_NULL_HANDLE;
    VkFence released = VK_NULL_HANDLE;
    bool pending = false;
};

inline void present_image_barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    auto access = [](VkImageLayout layout) -> VkAccessFlags {
        if (layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) return VK_ACCESS_TRANSFER_WRITE_BIT;
        if (layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) return VK_ACCESS_TRANSFER_READ_BIT;
        return 0;
    };
    b.srcAccessMask = access(from);
    b.dstAccessMask = access(to);
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         to == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &b);
}

inline void order_present_writes(VkCommandBuffer cmd, VkImage image) {
    present_image_barrier(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
}
// Both frame producers supply a GENERAL source and an optional view eligible for FSR.
VkRect2D picture_rect(VkExtent2D screen);
VkRect2D picture_crop(VkExtent2D source, VkRect2D requested);
enum class PicturePath { Clear, Blit, Fsr };
PicturePath compose_picture(VkCommandBuffer cmd, VkImage source, VkImageView fsr_view, VkExtent2D size, VkRect2D crop,
                     VkImage target, VkFormat format, VkExtent2D extent);
} // namespace nvn
