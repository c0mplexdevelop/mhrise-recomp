#include "nnsdk/nvn/present_image.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/fsr.h"
#include "nnsdk/nvn/resource_access.h"
#include <algorithm>
#include <cmath>

namespace nvn {
VkRect2D picture_rect(VkExtent2D screen) {
    float aspect = 16.0f / 9.0f;
    if (NVN_FLAG("aspect_4_3")) aspect = 4.0f / 3.0f;
    else if (NVN_FLAG("aspect_16_10")) aspect = 16.0f / 10.0f;
    else if (NVN_FLAG("aspect_21_9")) aspect = 21.0f / 9.0f;
    else if (NVN_FLAG("aspect_stretch")) aspect = 0.0f;
    VkRect2D rect{{0, 0}, screen};
    if (aspect <= 0.0f || !screen.width || !screen.height) return rect;
    const float shape = static_cast<float>(screen.width) / screen.height;
    if (shape > aspect + 0.01f) rect.extent.width = std::max(1u, static_cast<uint32_t>(std::lround(screen.height * aspect)));
    else if (shape < aspect - 0.01f) rect.extent.height = std::max(1u, static_cast<uint32_t>(std::lround(screen.width / aspect)));
    rect.offset = {static_cast<int32_t>((screen.width - rect.extent.width) / 2),
                   static_cast<int32_t>((screen.height - rect.extent.height) / 2)};
    return rect;
}
VkRect2D picture_crop(VkExtent2D source, VkRect2D crop) {
    crop.offset.x = std::clamp(crop.offset.x, 0, static_cast<int32_t>(source.width));
    crop.offset.y = std::clamp(crop.offset.y, 0, static_cast<int32_t>(source.height));
    const auto width = source.width - crop.offset.x, height = source.height - crop.offset.y;
    crop.extent.width = crop.extent.width ? std::min(crop.extent.width, width) : width;
    crop.extent.height = crop.extent.height ? std::min(crop.extent.height, height) : height;
    return crop;
}
PicturePath compose_picture(VkCommandBuffer cmd, VkImage source, VkImageView fsr_view, VkExtent2D size, VkRect2D crop,
                     VkImage target, VkFormat format, VkExtent2D extent) {
    const auto out = picture_rect(extent);
    constexpr VkImageSubresourceRange color{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (out.extent.width != extent.width || out.extent.height != extent.height) {
        const VkClearColorValue black{};
        vkCmdClearColorImage(cmd, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &color);
        order_present_writes(cmd, target);
    }
    if (fsr_view && fsr::record(cmd, fsr_view, size, crop.extent, target, format, out.extent, out.offset)) return PicturePath::Fsr;
    sync_image(cmd, source, color, prior_writes, transfer_read);
    VkImageBlit b{};
    b.srcSubresource = b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    b.srcOffsets[0] = {crop.offset.x, crop.offset.y, 0};
    b.srcOffsets[1] = {crop.offset.x + static_cast<int32_t>(crop.extent.width),
                       crop.offset.y + static_cast<int32_t>(crop.extent.height), 1};
    b.dstOffsets[0] = {out.offset.x, out.offset.y, 0};
    b.dstOffsets[1] = {out.offset.x + static_cast<int32_t>(out.extent.width),
                       out.offset.y + static_cast<int32_t>(out.extent.height), 1};
    vkCmdBlitImage(cmd, source, VK_IMAGE_LAYOUT_GENERAL, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_LINEAR);
    return PicturePath::Blit;
}
} // namespace nvn
