#pragma once
#include "nnsdk/nvn/present_image.h"

namespace nvn {
struct Texture;
struct Window;

struct PresentProbe {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    const uint8_t* mapped = nullptr;
    bool pending = false;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkFormat output_format = VK_FORMAT_UNDEFINED;
    bool detect = false;
    uint64_t time_ns = 0, buttons = 0;
    int sticks[4]{};
    nvn::PicturePath path{};

    void collect();
    VkImageLayout record(VkCommandBuffer cmd, Texture* source, VkRect2D crop, const Window& window,
                         VkImage target, PicturePath path);
};
}  // namespace nvn
