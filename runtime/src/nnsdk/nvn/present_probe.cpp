#include "nnsdk/nvn/present_probe.h"
#include "nnsdk/nvn/commands.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/resource_access.h"
#include "nnsdk/runtime.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace nvn {
namespace {
// --- Detector de destellos (DEPURACIÓN) ---
// <datos>/capture_flash (en vivo): el fuego del Aknosom salía como un disco amarillo plano que tapaba media pantalla
// durante ~0,5 s (5-11), demasiado poco para pedir la traza a mano. Cada frame se copian 16x9 texels de la textura de la
// ventana (dentro de su recorte) a un buffer de la CPU; al volver a ese juego de present (kPresentSlots frames después, con
// su fence ya esperado) se miran: con un sexto o más de amarillo saturado se pide la traza completa del frame siguiente
// (request_trace). Como mucho una cada 20 s y 4 por partida.
constexpr int kProbeW = 16, kProbeH = 9;

constexpr size_t kProbeStride = kProbeW * kProbeH * 8;


uint32_t texel_bytes(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return 4;
        case VK_FORMAT_R16G16B16A16_SFLOAT: return 8;
        default: return 0;
    }
}

float half_to_float(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    if (exp == 0) return (sign ? -1.0f : 1.0f) * man / 16777216.0f;
    if (exp == 31) return 65504.0f;
    return (sign ? -1.0f : 1.0f) * std::ldexp(1.0f + man / 1024.0f, static_cast<int>(exp) - 15);
}

bool make_probe(PresentProbe& p) {
    nvn::VkContext* c = nvn::vk();
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = 2 * kProbeStride;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(c->device, &bi, nullptr, &p.buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(c->device, p.buffer, &req);
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t type = UINT32_MAX;
    for (uint32_t k = 0; k < c->memory.memoryTypeCount && type == UINT32_MAX; k++)
        if ((req.memoryTypeBits >> k & 1) && (c->memory.memoryTypes[k].propertyFlags & want) == want) type = k;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    void* mapped = nullptr;
    if (type == UINT32_MAX || vkAllocateMemory(c->device, &ai, nullptr, &p.memory) != VK_SUCCESS ||
        vkBindBufferMemory(c->device, p.buffer, p.memory, 0) != VK_SUCCESS ||
        vkMapMemory(c->device, p.memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) {
        if (p.memory) vkFreeMemory(c->device, p.memory, nullptr);
        vkDestroyBuffer(c->device, p.buffer, nullptr);
        p = {};
        return false;
    }
    p.mapped = static_cast<const uint8_t*>(mapped);
    return true;
}

struct Rgb { float r, g, b; };
Rgb probe_rgb(VkFormat format, const uint8_t* q) {
    switch (format) {
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB: return {float(q[2]), float(q[1]), float(q[0])};
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: {
            uint32_t v;
            memcpy(&v, q, sizeof v);
            return {(v & 1023) / 4.f, ((v >> 10) & 1023) / 4.f, ((v >> 20) & 1023) / 4.f};
        }
        case VK_FORMAT_R16G16B16A16_SFLOAT: {
            uint16_t h[3];
            memcpy(h, q, sizeof h);
            return {std::clamp(half_to_float(h[0]), 0.f, 1.f) * 255,
                    std::clamp(half_to_float(h[1]), 0.f, 1.f) * 255,
                    std::clamp(half_to_float(h[2]), 0.f, 1.f) * 255};
        }
        default: return {float(q[0]), float(q[1]), float(q[2])};
    }
}

void log_layers(const PresentProbe& p) {
    uint64_t hash[2]{14695981039346656037ull, 14695981039346656037ull};
    unsigned dark[2]{}, darker = 0, brighter = 0;
    for (int k = 0; k < kProbeW * kProbeH; ++k) {
        float luma[2];
        for (int layer = 0; layer < 2; ++layer) {
            const auto format = layer ? p.output_format : p.format;
            const auto rgb = probe_rgb(format, p.mapped + layer * kProbeStride + k * texel_bytes(format));
            for (float v : {rgb.r, rgb.g, rgb.b}) hash[layer] = (hash[layer] ^ uint8_t(v)) * 1099511628211ull;
            luma[layer] = .2126f * rgb.r + .7152f * rgb.g + .0722f * rgb.b;
            dark[layer] += luma[layer] < 16;
        }
        darker += luma[0] - luma[1] > 32;
        brighter += luma[1] - luma[0] > 32;
    }
    NVN_LOG("present samples ns=%llu path=%s buttons=%llx sticks=%d,%d,%d,%d guest_rgb=%016llx window_rgb=%016llx "
            "dark=%u,%u darker=%u brighter=%u of %d\n", (unsigned long long)p.time_ns,
            p.path == nvn::PicturePath::Fsr ? "fsr" : "blit", (unsigned long long)p.buttons,
            p.sticks[0], p.sticks[1], p.sticks[2], p.sticks[3], (unsigned long long)hash[0],
            (unsigned long long)hash[1], dark[0], dark[1], darker, brighter, kProbeW * kProbeH);
}

}  // namespace

void PresentProbe::collect() {
    auto& p = *this;
    if (!p.pending) return;
    p.pending = false;
    if (p.output_format != VK_FORMAT_UNDEFINED) log_layers(p);
    if (!p.detect) return;
    const uint32_t bytes = texel_bytes(p.format);
    // Umbrales (en vivo, <datos>/destello_umbral: "R G Bmáx R-B sextos"): un destello cálido y muy claro. Los primeros
    // (230 190 200 40 y un sexto) no saltaron con uno casi blanco del 5-11; por defecto ahora 220 180 235 20.
    struct Thresholds {
        float r = 220, g = 180, b = 235, rb = 20;
        int sixths = 1;
    };
    static Thresholds t;
    static auto read_at = std::chrono::steady_clock::now() - std::chrono::seconds(10);
    if (std::chrono::steady_clock::now() - read_at > std::chrono::seconds(5)) {
        read_at = std::chrono::steady_clock::now();
        const char* data = getenv("NIER_DATA");
        if (FILE* f = fopen((std::string(data ? data : ".") + "/destello_umbral").c_str(), "r")) {
            Thresholds n;
            if (fscanf(f, "%f %f %f %f %d", &n.r, &n.g, &n.b, &n.rb, &n.sixths) == 5) t = n;
            fclose(f);
        }
    }
    int yellow = 0;
    for (int k = 0; k < kProbeW * kProbeH; k++) {
        const uint8_t* q = p.mapped + k * bytes;
        const auto [r, g, b] = probe_rgb(p.format, q);
        yellow += r >= t.r && g >= t.g && b <= t.b && r - b >= t.rb;
    }
    static auto last = std::chrono::steady_clock::now() - std::chrono::seconds(60);
    static int taken = 0;
    const auto now = std::chrono::steady_clock::now();
    if (yellow * 6 >= kProbeW * kProbeH * t.sixths && taken < 4 && now - last >= std::chrono::seconds(20)) {
        last = now;
        taken++;
        NVN_LOG("detector de destellos: %d de %d muestras en amarillo saturado: traza completa del frame siguiente\n", yellow,
                kProbeW * kProbeH);
        nvn::request_trace(true);
    }
}

// Graba la copia de las muestras de la textura de la ventana (en GENERAL, como la lee el blit).
static void record_samples(const PresentProbe& p, VkCommandBuffer cmd, VkImage image, VkImageLayout layout, VkFormat format,
                    VkRect2D crop, size_t offset) {
    const uint32_t bytes = texel_bytes(format), w = crop.extent.width, h = crop.extent.height;
    VkBufferImageCopy regions[kProbeW * kProbeH] = {};
    for (int y = 0; y < kProbeH; y++)
        for (int x = 0; x < kProbeW; x++) {
            VkBufferImageCopy& r = regions[y * kProbeW + x];
            r.bufferOffset = offset + static_cast<VkDeviceSize>(y * kProbeW + x) * bytes;
            r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            r.imageOffset = {crop.offset.x + static_cast<int32_t>((2 * x + 1) * w / (2 * kProbeW)),
                             crop.offset.y + static_cast<int32_t>((2 * y + 1) * h / (2 * kProbeH)), 0};
            r.imageExtent = {1, 1, 1};
        }
    vkCmdCopyImageToBuffer(cmd, image, layout, p.buffer, kProbeW * kProbeH, regions);
}

VkImageLayout PresentProbe::record(VkCommandBuffer cmd, Texture* src, VkRect2D crop, const Window& window,
                                    VkImage dst, PicturePath path) {
    auto& p = *this;
    constexpr auto written = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    const bool pair = window.capture_layers && NVN_FLAG("capture_layers");
    p.detect = NVN_FLAG("capture_flash");
    if (!pair && !p.detect) return written;
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (path == PicturePath::Clear) {
        if (pair) NVN_LOG("present samples ns=%llu path=clear\n", (unsigned long long)now);
        return written;
    }
    if (!texel_bytes(src->vk_format) || (!p.buffer && !make_probe(p))) return written;
    sync_image(cmd, src->image, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}, prior_writes, transfer_read);
    record_samples(p, cmd, src->image, VK_IMAGE_LAYOUT_GENERAL, src->vk_format, crop, 0);
    p.pending = true;
    p.format = src->vk_format;
    p.output_format = VK_FORMAT_UNDEFINED;
    auto final_layout = written;
    if (pair && texel_bytes(window.format)) {
        final_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        present_image_barrier(cmd, dst, written, final_layout);
        record_samples(p, cmd, dst, final_layout, window.format, picture_rect(window.extent), kProbeStride);
        p.output_format = window.format;
        p.time_ns = now;
        p.path = path;
        const auto* input = nnsdk::host().input;
        p.buttons = input ? input->buttons.load() : 0;
        for (int i = 0; i < 4; ++i) p.sticks[i] = input ? input->sticks[i].load() : 0;
    }
    sync_memory(cmd, transfer_write, {VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT});
    return final_layout;
}

}  // namespace nvn
