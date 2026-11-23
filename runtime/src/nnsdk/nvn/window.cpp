// nvnWindow*: la ventana NVN es un swapchain de Vulkan sobre la ventana de la APK. Las texturas
// de la ventana del juego (1280x720) son nuestras; al presentar se copian a la imagen del swapchain.
// Primer paso: se presenta el último ClearColor que el juego hizo sobre la textura.

#include <android/native_window.h>
#include <pthread.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstring>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "nnsdk/nvn/commands.h"
#include "nnsdk/nvn/driver.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/fsr.h"
#include "nnsdk/nvn/hud.h"
#include "nnsdk/nvn/lsfg/framegen.h"
#include "nnsdk/nvn/present_probe.h"
#include "nnsdk/nvn/nvn.h"
#include "nnsdk/nvn/objects.h"
#include "nnsdk/runtime.h"

namespace {

// NVNwindowBuilder es opaco; solo guardamos lo que necesitamos.
struct WindowBuilder {
    void* device;
    void* native_window;
    int present_interval;
    int texture_count;
    void* const* textures;
};

// Varios juegos de command buffer, fence y semáforos: con uno, presentar el frame N esperaba a que la GPU
// acabara el N-1 (CPU y GPU por turnos: la GPU al 66 % y 22 fps en la fábrica).
constexpr int kPresentSlots = 3;
struct PresentSlot {
    nvn::hud::Uploads hud;
    nvn::PresentProbe probe;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkSemaphore acquired = VK_NULL_HANDLE;
    VkFence done = VK_NULL_HANDLE;
    bool usable = true;
};
struct Presenter {
    VkCommandPool pool = VK_NULL_HANDLE;
    PresentSlot slots[kPresentSlots];
    int next = 0;
};

Presenter g_presenter;
// Un present cada vez (g_presenter, su pool y el ritmo). No es el del ejecutor: esperar al fence de hace
// kPresentSlots frames o a una imagen del swapchain no para a los hilos del juego ni a una reproducción.
std::mutex g_present_mutex;

bool init_presenter(nvn::VkContext& c) {
    if (g_presenter.pool) return true;
    Presenter next;
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = c.queue_family;
    VkResult r = vkCreateCommandPool(c.device, &pci, nullptr, &next.pool);
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = next.pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (auto& slot : next.slots) {
        if (r != VK_SUCCESS) break;
        r = vkAllocateCommandBuffers(c.device, &ai, &slot.cmd);
        if (r == VK_SUCCESS) r = vkCreateSemaphore(c.device, &si, nullptr, &slot.acquired);
        if (r == VK_SUCCESS) r = vkCreateFence(c.device, &fi, nullptr, &slot.done);
    }
    if (r == VK_SUCCESS) {
        g_presenter = std::move(next);
        return true;
    }
    NVN_LOG("present slots: %s\n", nvn::vk_result_name(r));
    for (auto& slot : next.slots) {
        vkDestroyFence(c.device, slot.done, nullptr);
        vkDestroySemaphore(c.device, slot.acquired, nullptr);
    }
    vkDestroyCommandPool(c.device, next.pool, nullptr);
    return false;
}

bool build_swapchain(nvn::Window& w) {
    nvn::VkContext* c = nvn::vk();
    if (!c || !init_presenter(*c)) return false;
    VkAndroidSurfaceCreateInfoKHR sci{VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR};
    sci.window = static_cast<ANativeWindow*>(w.native_window);
    if (VkResult r = vkCreateAndroidSurfaceKHR(c->instance, &sci, nullptr, &w.surface); r != VK_SUCCESS) {
        NVN_LOG("vkCreateAndroidSurfaceKHR: %s\n", nvn::vk_result_name(r));
        return false;
    }
    VkSurfaceCapabilitiesKHR caps;
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(c->gpu, w.surface, &caps) != VK_SUCCESS) return false;
    uint32_t count = 0;
    if (vkGetPhysicalDeviceSurfaceFormatsKHR(c->gpu, w.surface, &count, nullptr) != VK_SUCCESS || !count) return false;
    std::vector<VkSurfaceFormatKHR> formats(count);
    if (vkGetPhysicalDeviceSurfaceFormatsKHR(c->gpu, w.surface, &count, formats.data()) != VK_SUCCESS) return false;
    VkSurfaceFormatKHR format = formats[0];
    for (const VkSurfaceFormatKHR& f : formats)
        if (f.format == VK_FORMAT_R8G8B8A8_UNORM) format = f;

    // La pantalla de la Flip es vertical de fábrica (1080x1920) y Android la usa girada: el surface
    // da el tamaño ya en horizontal (1920x1080) con una rotación de 90°. Pretransformar obligaría a
    // dibujar girado (el blit no puede), así que se pide sin transformar y la gira el compositor.
    w.extent = caps.currentExtent;
    VkSurfaceTransformFlagBitsKHR transform = caps.currentTransform;
    if (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    NVN_LOG("swapchain %ux%u (transformación actual 0x%x, usada 0x%x)\n", w.extent.width, w.extent.height,
            caps.currentTransform, transform);
    VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    info.surface = w.surface;
    info.minImageCount = std::max(nvn::lsfg::swapchain_images(caps), caps.minImageCount);  // 4 con LSFG posible
    info.imageFormat = format.format;
    w.format = format.format;
    info.imageColorSpace = format.colorSpace;
    info.imageExtent = w.extent;
    info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    const bool capture_layers = NVN_FLAG("capture_layers");
    w.capture_layers = capture_layers && (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    if (w.capture_layers) info.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    else if (capture_layers) NVN_LOG("capture_layers: surface does not allow swapchain readback\n");
    info.preTransform = transform;
    info.compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    if (!(caps.supportedCompositeAlpha & info.compositeAlpha)) info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    info.clipped = VK_TRUE;
    if (VkResult r = vkCreateSwapchainKHR(c->device, &info, nullptr, &w.swapchain); r != VK_SUCCESS) {
        NVN_LOG("vkCreateSwapchainKHR: %s\n", nvn::vk_result_name(r));
        return false;
    }
    if (vkGetSwapchainImagesKHR(c->device, w.swapchain, &count, nullptr) != VK_SUCCESS || !count) return false;
    std::vector<VkImage> handles(count);
    if (vkGetSwapchainImagesKHR(c->device, w.swapchain, &count, handles.data()) != VK_SUCCESS) return false;
    w.images.resize(count);
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (uint32_t i = 0; i < count; ++i) {
        w.images[i].image = handles[i];
        if (VkResult r = vkCreateSemaphore(c->device, &si, nullptr, &w.images[i].ready); r != VK_SUCCESS) {
            NVN_LOG("present semaphore: %s\n", nvn::vk_result_name(r));
            return false;
        }
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (c->present_fences && vkCreateFence(c->device, &fi, nullptr, &w.images[i].released) != VK_SUCCESS) return false;
    }

    NVN_LOG("swapchain %ux%u, %u imágenes, formato %d, present fences %d\n",
            w.extent.width, w.extent.height, count, format.format, c->present_fences);
    return true;
}

void destroy_swapchain(nvn::Window& w) {
    auto* c = nvn::vk();
    if (!c) return;
    for (auto& image : w.images) {
        if (image.pending) vkWaitForFences(c->device, 1, &image.released, VK_TRUE, UINT64_MAX);
        vkDestroyFence(c->device, image.released, nullptr);
        vkDestroySemaphore(c->device, image.ready, nullptr);
    }
    w.images.clear();
    if (w.swapchain) vkDestroySwapchainKHR(c->device, w.swapchain, nullptr);
    if (w.surface) vkDestroySurfaceKHR(c->instance, w.surface, nullptr);
    w.swapchain = VK_NULL_HANDLE;
    w.surface = VK_NULL_HANDLE;
}

bool create_swapchain(nvn::Window& w) {
    if (build_swapchain(w)) return true;
    destroy_swapchain(w);
    return false;
}


}  // namespace

nvn::Objects<nvn::Window> nvn::g_windows;

VkResult nvn::Window::queue_image(uint32_t index, const VkPresentTimeGOOGLE* when) {
    auto* c = vk();
    auto& image = images[index];
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &image.ready;
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain;
    pi.pImageIndices = &index;
    VkPresentTimesInfoGOOGLE timing{VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE};
    if (when && c->display_timing) {
        timing.swapchainCount = 1;
        timing.pTimes = when;
        pi.pNext = &timing;
    }
    VkSwapchainPresentFenceInfoEXT fences{VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT};
    if (image.released) {
        if (image.pending) {
            VkResult r = vkWaitForFences(c->device, 1, &image.released, VK_TRUE, UINT64_MAX);
            if (r != VK_SUCCESS) return r;
            if ((r = vkResetFences(c->device, 1, &image.released)) != VK_SUCCESS) return r;
            image.pending = false;
        }
        fences.swapchainCount = 1;
        fences.pFences = &image.released;
        fences.pNext = pi.pNext;
        pi.pNext = &fences;
    }
    const VkResult result = vkQueuePresentKHR(c->queue, &pi);
    image.pending = image.released && (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR ||
                                       result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_ERROR_SURFACE_LOST_KHR);
    if (result < 0) rebuild = true;
    return result;
}

namespace {
// DEPURACIÓN: <datos>/lsfg_margen con unos milisegundos (p. ej. "4"): margen fijo entre el generado y el real, en vez
// del que se ajusta solo. Se lee como mucho una vez por segundo; -1 si no hay fichero.
int64_t live_margin_ns() {
    static int64_t value = -1;
    static std::chrono::steady_clock::time_point next{};
    const auto now = std::chrono::steady_clock::now();
    if (now >= next) {
        next = now + std::chrono::seconds(1);
        const char* data = getenv("NIER_DATA");
        double ms = -1;
        if (FILE* f = fopen((std::string(data ? data : ".") + "/lsfg_margen").c_str(), "r")) {
            if (fscanf(f, "%lf", &ms) != 1) ms = -1;
            fclose(f);
        }
        const int64_t v = ms < 0 ? -1 : static_cast<int64_t>(ms * 1e6);
        if (v != value) {
            if (v < 0) NVN_LOG("lsfg: margen del real tras el generado: automático\n");
            else NVN_LOG("lsfg: margen del real tras el generado fijo en %.1f ms\n", ms);
        }
        value = v;
    }
    return value;
}
// El real tiene que salir un refresco después de que salga el generado, y el generado sale cuando la GPU lo acaba
// (detrás del frame del juego): desde que se manda hasta que se ve pasan L ms, que dependen de la carga. Se mide L con
// los tiempos de presentación pasados (VK_GOOGLE_display_timing) y el real se pide para "mandado + 16,7 + L - 1"; si
// un generado no llega a verse (el compositor lo tiró), se sube L.
class LsfgPacer {
public:
    int64_t margin_ns(VkDevice device, VkSwapchainKHR swapchain) {
        if (const int64_t fixed = live_margin_ns(); fixed >= 0) return fixed;
        if (!get_past_) get_past_ = reinterpret_cast<PFN_vkGetPastPresentationTimingGOOGLE>(
                            vkGetDeviceProcAddr(device, "vkGetPastPresentationTimingGOOGLE"));
        if (get_past_) {
            uint32_t n = 0;
            get_past_(device, swapchain, &n, nullptr);
            n = std::min<uint32_t>(n, 16);
            VkPastPresentationTimingGOOGLE past[16];
            if (n && get_past_(device, swapchain, &n, past) >= 0)
                for (uint32_t k = 0; k < n; k++) note(past[k]);
        }
        return std::max<int64_t>(latency_ns_ - 1000000, 2000000);
    }
    void queued_generated(uint32_t id, int64_t ns) { sent_[id % kRing] = {id, ns}; }

private:
    static constexpr uint32_t kRing = 64;
    struct Sent {
        uint32_t id = 0;
        int64_t ns = 0;
    };
    void note(const VkPastPresentationTimingGOOGLE& t) {
        if (!(t.presentID & 1)) {  // un real: ¿se vio el generado de delante?
            const uint32_t g = t.presentID - 1;
            if (g > last_seen_g_ && g > 1) {
                latency_ns_ = std::min<int64_t>(latency_ns_ + 4000000, 45000000);  // no salió: más margen
                if (dropped_++ < 5) NVN_LOG("lsfg: el generado %u no se vio; margen %.1f ms\n", g, latency_ns_ / 1e6);
            }
            return;
        }
        const Sent& s = sent_[t.presentID % kRing];
        if (s.id != t.presentID || !t.actualPresentTime) return;
        last_seen_g_ = t.presentID;
        const int64_t l = static_cast<int64_t>(t.actualPresentTime) - s.ns;
        if (l > 0 && l < 100000000) latency_ns_ += (l - latency_ns_) / 8;  // media móvil
    }
    PFN_vkGetPastPresentationTimingGOOGLE get_past_ = nullptr;
    Sent sent_[kRing];
    uint32_t last_seen_g_ = 0, dropped_ = 0;
    int64_t latency_ns_ = 12000000;
};
LsfgPacer g_lsfg_pacer;

// El reloj de MH Rise (16.0.1), en el objeto de via.Application (App = *(main+0x12950348)). Su bucle principal
// (main+0x6b047c0) mide cada vuelta con GetSystemTick, ejecuta el frame (main+0x6b03fa4) y luego ESPERA (sleep de 1 ms,
// y girando los últimos 2 ms) hasta que hayan pasado App+0x430 / App+0x434 frames de 60 Hz: base 60 y MaxFps 30 en
// juego (main+0x3600f04 pone MaxFps 60 u 30 y el intervalo 1 o 2 de la ventana a la vez; las cargas van a 60). El paso es
// variable (App+0x441, isFixedFrameRate, a 0): dt = lo medido en frames de 60 Hz (como mucho 8, App+0x438; más de 50,
// App+0x43c, vuelve a 1) en App+0x41c, y en segundos en App+0x424; App+0x420 = ms de la vuelta anterior. O sea: entre
// 7,5 y 60 fps el juego va a su velocidad (no a cámara lenta como NieR), pero forzar el intervalo 1 al presentar no
// basta para 60: el limitador del juego seguía en 30. Con <datos>/60fps se le pone MaxFps 60 en cada present (el juego lo
// vuelve a poner en 30 al cambiar de escena) y al quitarlo se le devuelve lo suyo.
constexpr size_t kAppPointer = 0x12950348;      // via.Application* (dato de main)
constexpr size_t kLimiterLoad = 0x6b04828;      // ldr s1, [x19, #0x434]: el MaxFps que lee el limitador
constexpr uint32_t kLimiterLoadCode = 0xBD443661;
struct GameClock {  // App+0x41c..0x434
    float dt60, frame_ms, dt, unused[2], base, max_fps;
};
static_assert(sizeof(GameClock) == 0x434 + 4 - 0x41c);

GameClock* game_clock() {
    static const uint8_t* base = nnsdk::host().main_base;
    static const bool known = [] {
        const bool ok = base && nnsdk::host().main_size > kAppPointer + 8 &&
                        *reinterpret_cast<const uint32_t*>(base + kLimiterLoad) == kLimiterLoadCode;
        NVN_LOG("reloj del juego: %s\n", ok ? "MH Rise 16.0.1 (App en main+0x12950348)" : "no es el main conocido: sin tocar MaxFps");
        return ok;
    }();
    if (!known) return nullptr;
    const uint64_t app = *reinterpret_cast<const volatile uint64_t*>(base + kAppPointer);
    if (!app) return nullptr;  // aún sin via.Application
    auto* c = reinterpret_cast<GameClock*>(app + 0x41c);
    return c->base == 60.0f ? c : nullptr;
}

// --- Resolución dinámica (MH Rise 16.0.1) ---
// via.render.Renderer (*(main+0x12950538)) guarda en +0x1a78 su ImageQualityRate. La actualización de RenderOutput
// (main+0x7936a24) lo multiplica cada frame por su ImageQuality (en portátil la escena sale a 960x540) y layer.Scene recorta
// a eso su región (ImageQualityRegion) y la escala él a la salida: el resto del port no cambia. El juego solo lo toca al
// cargar (main+0x5389b50: 0,1 mientras carga, 1,0 al acabar). Su propia resolución dinámica (snow.DynamicResolutionManager)
// escribe en RenderOutput y aquí no baja de 960x540; esto multiplica encima. Cambiarlo cuesta un frame sin la historia del
// TAA: por pasos (anchos múltiplos de 64: 960, 896, 832, 768, 704) y con esperas. En combate la GPU del frame del juego
// pasaba de 33 ms (al 90-99 % y a su reloj máximo): 25 y 21 fps.
// Baja un paso cuando la media de los últimos 8 frames pasa del 91 % del presupuesto (33,3 ms a 30 fps) o dos de ellos
// pasan del 100 %, como pronto 0,5 s después del cambio anterior y solo si el ejecutor cabe (si el frame lo limita la CPU,
// bajar la resolución no lo arregla). Sube tras 3 s sin cambios si el peor de esos 90 frames, escalado por los píxeles
// del paso de arriba, cabe en el 78 % (con 2 s y el 80 % subía y bajaba 7 veces en 10 s al entrar en la misión).
// <datos>/no_dynamic_resolution (en vivo): quieta en la resolución base.
// Resolución base (en vivo, el menú de opciones): la escena de portátil es 960x540 dentro de la salida de 1280x720, y la
// profundidad y el resto de la escena ya son de 1280x720, así que el ratio puede subir hasta llenarla. <datos>/resolution_576,
// resolution_648 o resolution_720: la escena a 1024x576, 1152x648 o 1280x720 (720p de verdad, sin escalar del juego), y
// la dinámica baja desde ahí con los mismos pasos. Sin ninguno, 960x540 como el juego.
constexpr size_t kRendererPointer = 0x12950538;  // via.render.Renderer* (dato de main)
constexpr size_t kRateField = 0x1a78;
constexpr size_t kRateSetter = 0x78a0ea0;  // set_ImageQualityRate: adrp/ldr del Renderer y str s0, [x8, #0x1a78]
constexpr uint32_t kRateSetterCode[3] = {0x90058588, 0xF9429D08, 0xBD1A7900};
constexpr size_t kRateLoad = 0x7936e20;  // ldr s1, [x8, #0x1a78] en la actualización de RenderOutput
constexpr uint32_t kRateLoadCode = 0xBD5A7901;
constexpr float kRates[] = {1.0f, 0.93334f, 0.86667f, 0.80001f, 0.73334f};  // un poco por encima: floor no pierde un píxel
constexpr int kLevels = static_cast<int>(std::size(kRates));

// La base elegida: el ratio sobre 960x540 y las líneas, para el log
struct BaseResolution {
    float rate;
    int lines;
};
BaseResolution base_resolution() {
    if (NVN_FLAG("resolution_720")) return {1.33334f, 720};
    if (NVN_FLAG("resolution_648")) return {1.20001f, 648};
    if (NVN_FLAG("resolution_576")) return {1.06668f, 576};
    // Lower than the game's: for weak GPUs (the dynamic resolution still steps down from there, never under 0.2,
    // which the game uses while loading)
    if (NVN_FLAG("resolution_480")) return {0.88890f, 480};
    if (NVN_FLAG("resolution_432")) return {0.80001f, 432};
    if (NVN_FLAG("resolution_360")) return {0.66668f, 360};
    if (NVN_FLAG("resolution_240")) return {0.44445f, 240};
    return {1.0f, 540};
}

struct DynamicResolution {
    int level = 0;
    int since_change = 1000;  // frames de GPU desde el último cambio
    int64_t samples[90] = {};
    int count = 0, next = 0;
    uint64_t serial = 0;
    uint64_t changes = 0;   // para el log de ritmo
    bool written = false;   // se escribió algo distinto de 1,0
    int logged = 0;
};
DynamicResolution g_dynres;

volatile float* game_quality_rate() {
    static const uint8_t* base = nnsdk::host().main_base;
    static const bool known = [] {
        const auto* code = reinterpret_cast<const uint32_t*>(base + kRateSetter);
        const bool ok = base && nnsdk::host().main_size > kRendererPointer + 8 && code[0] == kRateSetterCode[0] &&
                        code[1] == kRateSetterCode[1] && code[2] == kRateSetterCode[2] &&
                        *reinterpret_cast<const uint32_t*>(base + kRateLoad) == kRateLoadCode;
        NVN_LOG("resolución dinámica: %s\n", ok ? "MH Rise 16.0.1 (Renderer en main+0x12950538, ImageQualityRate en +0x1a78)"
                                                : "no es el main conocido: sin tocar la resolución");
        return ok;
    }();
    if (!known) return nullptr;
    const uint64_t renderer = *reinterpret_cast<const volatile uint64_t*>(base + kRendererPointer);
    return renderer ? reinterpret_cast<volatile float*>(renderer + kRateField) : nullptr;
}

// Desde el hilo que presenta, una vez por frame. period_ns: lo que dura un frame (el presupuesto).
void dynamic_resolution(int64_t period_ns) {
    DynamicResolution& d = g_dynres;
    volatile float* rate = game_quality_rate();
    if (!rate) return;
    const float now = *rate;
    if (now < 0.2f) {  // cargando (0,1): es del juego; al acabar pone 1,0 y se vuelve a empezar desde arriba
        d.level = 0;
        d.count = d.next = 0;
        d.since_change = 1000;
        return;
    }
    const BaseResolution base = base_resolution();
    static int base_logged = 0;
    if (base.lines != base_logged) {
        NVN_LOG("resolución base: %dp (escena %dx%d)%s\n", base.lines, static_cast<int>(960 * base.rate),
                static_cast<int>(540 * base.rate), NVN_FLAG("no_dynamic_resolution") ? ", fija" : ", la dinámica baja desde ahí");
        base_logged = base.lines;
    }
    if (NVN_FLAG("no_dynamic_resolution")) {
        d.level = 0;
        if (now != base.rate) *rate = base.rate;
        d.written = base.rate != 1.0f;
        return;
    }
    int64_t gpu = 0;
    const bool fresh = nvn::game_gpu_frame(d.serial, gpu);
    if (fresh && gpu > 0) {
        d.samples[d.next] = gpu;
        d.next = (d.next + 1) % 90;
        d.count = std::min(d.count + 1, 90);
        d.since_change++;
        const double budget = static_cast<double>(period_ns);
        auto last = [&](int k) { return d.samples[(d.next - 1 - k + 180) % 90]; };
        int level = d.level;
        if (d.count >= 8 && d.since_change >= 15 && level + 1 < kLevels) {
            double sum = 0;
            int over = 0;
            for (int k = 0; k < 8; k++) sum += last(k), over += last(k) > budget;
            const bool cpu_fits = nvn::last_frame_run_ns() < 0.85 * budget;
            if ((sum / 8 > 0.91 * budget || over >= 2) && cpu_fits) level++;
        }
        if (level == d.level && d.count >= 90 && d.since_change >= 90 && level > 0) {
            int64_t worst = 0;
            for (int k = 0; k < 90; k++) worst = std::max(worst, last(k));
            const double up = static_cast<double>(kRates[level - 1]) / kRates[level];
            if (worst * up * up < 0.78 * budget) level--;
        }
        if (level != d.level) {
            if (d.logged++ < 40)
                NVN_LOG("resolución dinámica: x%.3f -> x%.3f (escena ~%dx%d), GPU del juego %.1f ms\n", kRates[d.level],
                        kRates[level], static_cast<int>(960 * base.rate * kRates[level]),
                        static_cast<int>(540 * base.rate * kRates[level]), gpu / 1e6);
            d.level = level;
            d.since_change = 0;
            d.changes++;
        }
    }
    const float want = base.rate * kRates[d.level];
    if (now != want) {
        *rate = want;
        d.written = d.written || want != 1.0f;
    }
}

// Registro de ritmo (siempre, sin <datos>/verbose_log): cada 10 s, si hubo algún frame de más de 40 ms (por debajo de 30 fps,
// con el intervalo 2 del juego), una línea con por qué pudo ser: la GPU del frame del juego (media y peor), el ejecutor
// por frame (y lo que esperó a la GPU), las pipelines que se crearon en NvnSubmit con el frame parado, las que fueron al
// hilo de segundo plano (y los draws que no se dibujaron esperándolas) y las traducciones en NvnSubmit y en la precarga.
// Sin frames lentos, una cada 60 s. Con fflush: si el juego se cierra, no se pierde lo último (va por la tubería, no a la
// SD desde aquí).
void log_pace(const nvn::Window& w) {
    using clock = std::chrono::steady_clock;
    static clock::time_point window_start = clock::now(), last_present = window_start, last_line = window_start;
    static uint64_t frames = 0, slow = 0, worst_ns = 0;
    const clock::time_point now = clock::now();
    const uint64_t gap = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - last_present).count());
    last_present = now;
    frames++;
    if (gap > 40'000'000) slow++;
    worst_ns = std::max(worst_ns, gap);
    const double secs = std::chrono::duration<double>(now - window_start).count();
    if (secs < 10.0) return;
    nvn::DropStats& d = nvn::g_drop_stats;
    auto take = [](std::atomic<uint64_t>& a) { return a.exchange(0, std::memory_order_relaxed); };
    const uint64_t gpu_sum = take(d.gpu_sum_ns), gpu_max = take(d.gpu_max_ns), gpu_frames = take(d.gpu_frames);
    const uint64_t run = take(d.run_ns), run_max = take(d.run_max_ns), wait = take(d.wait_ns);
    const uint64_t live = take(d.pipelines_live), live_ns = take(d.pipelines_live_ns);
    const uint64_t async = take(d.pipelines_async), skipped = take(d.draws_skipped);
    const uint64_t pipeline_wait = take(d.pipeline_wait_ns);
    const uint64_t known = take(d.pipelines_known), known_ns = take(d.pipelines_known_ns);
    const uint64_t translated = take(d.translated_live), translated_ns = take(d.translated_live_ns);
    const uint64_t prefetched = take(d.prefetched), prefetch_ns = take(d.prefetch_ns);
    std::string slowest;
    uint64_t slowest_ns = 0;
    {
        std::lock_guard lock(d.slow_mutex);
        slowest_ns = std::exchange(d.slow_ns, 0);
        slowest = std::move(d.slow_summary);
        d.slow_summary.clear();
    }
    if (slow || now - last_line >= std::chrono::seconds(60)) {
        last_line = now;
        NVN_LOG("ritmo: %.1f fps, %llu frames de más de 40 ms (el peor %.0f ms), intervalo %d, resolución x%.3f (%llu cambios); "
                "GPU del juego %.1f ms (peor "
                "%.1f); ejecutor %.1f ms/frame (peor %.1f, %.1f esperando a la GPU); pipelines en NvnSubmit %llu (%.0f ms), "
                "en segundo plano %llu (%.1f ms esperando pipelines), %llu draws omitidos por traducción, registradas %llu (%.0f ms); traducciones en NvnSubmit %llu (%.0f ms), precargadas %llu "
                "(%.0f ms)\n",
                frames / secs, static_cast<unsigned long long>(slow), worst_ns / 1e6, w.present_interval, kRates[g_dynres.level],
                static_cast<unsigned long long>(std::exchange(g_dynres.changes, 0)),
                gpu_frames ? gpu_sum / 1e6 / gpu_frames : 0.0, gpu_max / 1e6, run / 1e6 / std::max<uint64_t>(frames, 1),
                run_max / 1e6, wait / 1e6 / std::max<uint64_t>(frames, 1), static_cast<unsigned long long>(live), live_ns / 1e6,
                static_cast<unsigned long long>(async), pipeline_wait / 1e6, static_cast<unsigned long long>(skipped),
                static_cast<unsigned long long>(known), known_ns / 1e6, static_cast<unsigned long long>(translated), translated_ns / 1e6, static_cast<unsigned long long>(prefetched),
                prefetch_ns / 1e6);
        if (slowest_ns) NVN_LOG("ritmo: el envío más lento, %.0f ms: %s\n", slowest_ns / 1e6, slowest.c_str());
        fflush(stdout);
    }
    window_start = now;
    frames = slow = worst_ns = 0;
}

// Límite de FPS (el menú de opciones; en vivo): <datos>/fps_20, fps_25, fps_40, fps_45 o 60fps. Sin ninguno, lo del juego
// (30 jugando, 60 en las cargas). El paso del juego es variable (ver GameClock): entre 7,5 y 60 fps va a su velocidad, así
// que bajar el límite ahorra GPU y batería sin cámara lenta; subirlo pide más. En las cargas el juego pide 60 y se le deja
// (un límite bajo las alargaría).
int fps_cap() {
    if (NVN_FLAG("60fps")) return 60;
    if (NVN_FLAG("fps_45")) return 45;
    if (NVN_FLAG("fps_40")) return 40;
    if (NVN_FLAG("fps_25")) return 25;
    if (NVN_FLAG("fps_20")) return 20;
    return 0;
}

// Pone el MaxFps del juego para ese límite (0: el suyo, de vuelta si se había cambiado). Devuelve los fps a los que va
// el juego, o 0 si no se conoce su reloj. Desde el hilo que presenta.
float game_max_fps(int cap) {
    static float wanted = 30.0f;  // lo último que puso el juego
    static float ours = 0.0f;     // lo que pusimos nosotros (0: nada)
    static float logged = 0.0f;
    GameClock* c = game_clock();
    if (!c) return 0.0f;
    volatile float& max_fps = c->max_fps;
    const float now = max_fps;
    if (ours == 0.0f || now != ours) wanted = now;  // lo ha cambiado el juego
    float target = wanted;
    if (cap) {
        target = wanted >= 60.0f ? 60.0f : static_cast<float>(cap);
        if (now != target) max_fps = target;
        ours = target;
    } else if (ours != 0.0f) {
        if (now == ours) max_fps = wanted;
        ours = 0.0f;
    }
    if (target != logged) {
        NVN_LOG("reloj del juego: MaxFps %.0f%s\n", target, cap ? " (límite del menú)" : "");
        logged = target;
    }
    return target;
}
}  // namespace

// La ventana de Android cambió (la app se fue a segundo plano o volvió; ver NnsdkWindow): se suelta el swapchain y la
// superficie de la vieja y, si hay una nueva, se crean en ella. Antes, al volver del escritorio la pantalla se quedaba
// en negro (se seguía presentando en la ventana destruida).
void follow_native_window(nvn::Window& w) {
    NnsdkWindow* state = nnsdk::host().window_state;
    const uint32_t serial = state ? state->serial.load(std::memory_order_acquire) : 0;
    if (!w.rebuild && (!state || serial == state->done.load(std::memory_order_relaxed))) return;
    void* now = state ? state->window.load(std::memory_order_acquire) : w.native_window;
    nvn::VkContext* c = nvn::vk();
    if (now != w.native_window || !w.swapchain || w.rebuild) {
        {
            std::lock_guard queue_lock(nvn::vk_queue_mutex());
            vkQueueWaitIdle(c->queue);  // lo presentado en la vieja, terminado antes de soltarla
        }
        destroy_swapchain(w);
        w.rebuild = false;
        w.native_window = now;
        if (now) {
            const bool ok = create_swapchain(w);
            w.rebuild = !ok;
            NVN_LOG("ventana nueva: swapchain %s\n", ok ? "creado" : "sin crear");
        } else {
            NVN_LOG("sin ventana (segundo plano): no se presenta\n");
        }
    }
    if (state) state->done.store(serial, std::memory_order_release);
}

namespace {
std::atomic<int64_t> g_frame_period_ns{33'333'333};
}  // namespace

int64_t nvn::frame_period_ns() { return g_frame_period_ns.load(std::memory_order_relaxed); }

namespace {
// The picture on screen: frame generation's pictures, then the real one (blit to a swapchain image and present). On
// NvnSubmit after the pacing, or with asynchronous presentation on the Presenter thread (see nvn::present).
void show(nvn::Window& w, nvn::ResourceRef<nvn::Texture> source) {
    using namespace nvn;
    VkContext* c = vk();
    std::lock_guard present_lock(g_present_mutex);
    if (!w.swapchain) return;
    fsr::release_if_resized(picture_rect(w.extent).extent);
    hud::real_frame(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    // LSFG 2x encendido: antes del real, el frame intermedio (lsfg/framegen.cpp).
    uint32_t generated_id = 0;
    const int64_t generated_ns = lsfg::present_generated(w, source, &generated_id);
    if (w.rebuild) return;
    if (generated_ns) g_lsfg_pacer.queued_generated(generated_id, generated_ns);
    PresentSlot& slot = g_presenter.slots[g_presenter.next];
    g_presenter.next = (g_presenter.next + 1) % kPresentSlots;

    if (!slot.usable) return;
    VkResult r = vkWaitForFences(c->device, 1, &slot.done, VK_TRUE, UINT64_MAX);
    if (r != VK_SUCCESS) {
        NVN_LOG("present slot fence: %s\n", vk_result_name(r));
        return;
    }
    // Las muestras de cualquier juego de present ya terminado, no solo las de este (el de hace kPresentSlots frames): el
    // del frame anterior suele estarlo ya, y así la traza cae 2 frames después del destello y no 4 (el 5-11 llegó tarde:
    // en el frame trazado el cazador ya estaba en el suelo).
    for (auto& queued : g_presenter.slots)
        if (queued.probe.pending && (&queued == &slot || vkGetFenceStatus(c->device, queued.done) == VK_SUCCESS))
            queued.probe.collect();
    uint32_t image = 0;
    r = vkAcquireNextImageKHR(c->device, w.swapchain, UINT64_MAX, slot.acquired, VK_NULL_HANDLE, &image);
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_ERROR_SURFACE_LOST_KHR) w.rebuild = true;
        NVN_LOG("vkAcquireNextImageKHR: %s\n", vk_result_name(r));
        return;  // el fence sigue señalado (se reinicia después): este contexto no se queda esperando para siempre
    }
    // On a failed recording or submit, drain the acquisition before using this slot again.
    auto abandon = [&](VkResult error) {
        NVN_LOG("present submission: %s\n", vk_result_name(error));
        w.rebuild = true;
        slot.probe.pending = false;
        VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo wait{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        wait.waitSemaphoreCount = 1;
        wait.pWaitSemaphores = &slot.acquired;
        wait.pWaitDstStageMask = &stage;
        std::lock_guard lock(vk_queue_mutex());
        slot.usable = vkResetFences(c->device, 1, &slot.done) == VK_SUCCESS &&
                      vkQueueSubmit(c->queue, 1, &wait, slot.done) == VK_SUCCESS;
    };

    // El blit lee la imagen de la textura: desde grabarlo hasta enviarlo, con el lock del ejecutor. Así, si el
    // juego la retira, se libera tras un envío del ejecutor posterior a este (va detrás en la cola).
    std::unique_lock executor_lock(executor_mutex());
    Texture* t = source.get();
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if ((r = vkResetCommandBuffer(slot.cmd, 0)) != VK_SUCCESS ||
        (r = vkBeginCommandBuffer(slot.cmd, &bi)) != VK_SUCCESS) return abandon(r);
    const auto& target = w.images[image];
    present_image_barrier(slot.cmd, target.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkClearColorValue color{};
    if (t && t->cleared) memcpy(color.float32, t->clear_color, sizeof color.float32);
    // Lo que el juego dibujó en la textura de la ventana, escalado a la pantalla; si aún no hay
    // imagen (no se ha dibujado nada), su último color de borrado.
    const auto crop = t ? picture_crop({uint32_t(t->desc.width), uint32_t(t->desc.height)}, w.crop) : VkRect2D{};
    const auto path = t ? blit_to_swapchain(*t, crop, target.image, w.format, w.extent, slot.cmd) : PicturePath::Clear;
    if (path == PicturePath::Clear) {
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(slot.cmd, target.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
    }
    hud::record(slot.hud, slot.cmd, target.image, w.extent, w.format);
    const auto final_layout = slot.probe.record(slot.cmd, t, crop, w, target.image, path);
    present_image_barrier(slot.cmd, target.image, final_layout, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    if ((r = vkEndCommandBuffer(slot.cmd)) != VK_SUCCESS ||
        (r = vkResetFences(c->device, 1, &slot.done)) != VK_SUCCESS) return abandon(r);

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &slot.acquired;
    si.pWaitDstStageMask = &wait_stage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &slot.cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &target.ready;
    {
        std::lock_guard queue_lock(vk_queue_mutex());
        r = vkQueueSubmit(c->queue, 1, &si, slot.done);
    }
    if (r != VK_SUCCESS) return abandon(r);
    executor_lock.unlock();

    // Con un generado delante, el real no antes de un refresco después: con los dos listos en el mismo refresco,
    // el compositor de Android se queda con el más nuevo y tira el otro (en pantalla seguían saliendo 30).
    // <datos>/lsfg_margen (ms) añade margen en vivo por si el generado aún no está en pantalla a esa hora.
    VkPresentTimeGOOGLE when{};
    if (generated_ns && c->display_timing) {
        when.presentID = generated_id + 1;  // par: el real detrás del generado generated_id
        when.desiredPresentTime =
            static_cast<uint64_t>(generated_ns + lsfg::spacing_ns() + g_lsfg_pacer.margin_ns(c->device, w.swapchain));
    }
    {
        std::lock_guard queue_lock(vk_queue_mutex());
        r = w.queue_image(image, generated_ns ? &when : nullptr);
    }
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
        NVN_LOG("vkQueuePresentKHR: %s\n", vk_result_name(r));
        return;
    }
    w.presented++;
    if (w.presented == 60) nvn::driver::works();  // (un driver del usuario que ya ha dibujado: se queda elegido)
}

// Asynchronous presentation (Settings → Performance, async_present; Eden's option of the same name): NvnSubmit hands the
// frame to this thread after its pacing and goes on with the next frame's command lists while this one waits for a
// swapchain image, makes the generated pictures and presents. One frame at most: NvnSubmit waits for the last one before
// it paces the next (so the game's window texture is never drawn over before its blit is on the queue).
struct AsyncPresenter {
    std::mutex m;
    std::condition_variable cv;
    nvn::Window* window = nullptr;
    nvn::ResourceRef<nvn::Texture> texture;
    bool busy = false, started = false;

    void wait_idle() {
        std::unique_lock lock(m);
        cv.wait(lock, [&] { return !busy; });
    }
    void post(nvn::Window& w, nvn::ResourceRef<nvn::Texture> t) {
        std::lock_guard lock(m);
        if (!started) {
            started = true;
            std::thread([this] { loop(); }).detach();  // (lives as long as the process)
        }
        window = &w;
        texture = t;
        busy = true;
        cv.notify_all();
    }
    void loop() {
        pthread_setname_np(pthread_self(), "Presenter");
        nvn::pin_critical_thread();
        std::unique_lock lock(m);
        for (;;) {
            cv.wait(lock, [&] { return window != nullptr; });
            nvn::Window* w = std::exchange(window, nullptr);
            const auto t = texture;
            lock.unlock();
            show(*w, t);
            lock.lock();
            busy = false;
            cv.notify_all();
        }
    }
};
AsyncPresenter g_presenter_thread;
}  // namespace

void nvn::present(Window& w, int index) {
    VkContext* c = vk();
    if (!c || index < 0 || index >= static_cast<int>(w.textures.size())) return;
    g_presenter_thread.wait_idle();  // (the last frame's present, with asynchronous presentation: done first)
    ResourceRef<Texture> t;
    {
        std::lock_guard present_lock(g_present_mutex);
        follow_native_window(w);
        t = w.textures[index];
    }
    // Intervalo de presentación de NVN (NieR pide 2: 30 fps, como en la Switch). FIFO solo garantiza
    // uno por vsync, así que el resto lo pone este ritmo. El reloj del juego mide tiempo real (pasos de 1/60 s,
    // entre 1 y 2 por frame): a 60 va a su velocidad, y por debajo de 30 a cámara lenta (eso es NieR; MH Rise mide su
    // paso, ver GameClock). Con <datos>/60fps se fuerza el intervalo 1 (y, en MH Rise, MaxFps 60). Sin el lock del
    // ejecutor: dormido con él, los hilos del juego que esperaban un sync se quedaban parados hasta 33 ms.
    {
        using clock = std::chrono::steady_clock;
        static clock::time_point next = clock::now();
        static int64_t logged_period = 0;
        const int cap = fps_cap();
        const float game_fps = game_max_fps(cap);
        // Con límite, el frame dura lo que pide (lo que el juego tenga de MaxFps; sin su reloj, el límite tal cual); sin él,
        // el intervalo de NVN que pide el juego (x 16,7 ms)
        const float fps = cap ? (game_fps > 0.0f ? game_fps : static_cast<float>(cap)) : 0.0f;
        const int64_t period_ns = fps > 0.0f ? static_cast<int64_t>(1e9 / fps) : 16'666'667LL * std::max(w.present_interval, 1);
        g_frame_period_ns.store(period_ns, std::memory_order_relaxed);
        dynamic_resolution(period_ns);
        game_tweaks();
        if (period_ns != logged_period) {
            NVN_LOG("ritmo de presentación: %.1f ms por frame (%.0f fps)%s\n", period_ns / 1e6, 1e9 / period_ns,
                    cap ? " (límite del menú)" : "");
            logged_period = period_ns;
        }
        const auto period = std::chrono::nanoseconds(period_ns);
        // Con frames de más de 20 ms se despierta 2 ms antes (FIFO pone el resto). A 60, en `next`: un frame de menos de
        // 16,7 ms le cuenta al juego como uno entero y lo aceleraría un poco.
        const auto early = std::chrono::milliseconds(period_ns > 20'000'000 ? 2 : 0);
        auto now = clock::now();
        if (now < next - early) std::this_thread::sleep_until(next - early);
        next = std::max(next + period, clock::now() + period / 2);
    }
    capture_loading_present(w.swapchain ? t : nullptr, w.crop);
    if (!w.swapchain) {  // sin ventana (en segundo plano): el juego sigue a su ritmo, sin presentar
        std::lock_guard executor_lock(executor_mutex());
        next_frame();
        return;
    }
    if (NVN_FLAG("async_present")) {
        g_presenter_thread.post(w, t);
    } else {
        show(w, t);
    }
    {
        std::lock_guard executor_lock(executor_mutex());  // el frame del ejecutor y sus volcados
        debug_dump_frame(t);
        next_frame();
    }
    log_pace(w);
    // Rendimiento: frames por segundo cada ~5 s.
    static auto window_start = std::chrono::steady_clock::now();
    static uint64_t window_frames = 0;
    window_frames++;
    auto now = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(now - window_start).count();
    if (secs >= 5.0) {
        if (NVN_FLAG("verbose_log")) {
            NVN_LOG("%.1f fps (frame %llu)\n", window_frames / secs, static_cast<unsigned long long>(w.presented));
            // El reloj del juego: su MaxFps, lo que midió de la última vuelta y el paso que dio (a 60 fps, ~16,7 ms y 1).
            if (const GameClock* c = game_clock())
                NVN_LOG("reloj del juego: MaxFps %.0f, vuelta %.1f ms, paso %.2f frames de 60 Hz (%.4f s)\n", c->max_fps,
                        c->frame_ms, c->dt * 60.0f, c->dt);
        }
        window_start = now;
        window_frames = 0;
    }
}

extern "C" {

void nvnWindowBuilderSetDefaults(WindowBuilder* b) {
    memset(b, 0, sizeof *b);
    b->present_interval = 1;
}
void nvnWindowBuilderSetDevice(WindowBuilder* b, void* device) { b->device = device; }
void nvnWindowBuilderSetNativeWindow(WindowBuilder* b, void* native_window) { b->native_window = native_window; }
void nvnWindowBuilderSetPresentInterval(WindowBuilder* b, int interval) { b->present_interval = interval; }
void nvnWindowBuilderSetTextures(WindowBuilder* b, int count, void* const* textures) {
    b->texture_count = count;
    b->textures = textures;
}

nvn::Boolean nvnWindowInitialize(void* window, const WindowBuilder* b) {
    nvn::Window& w = nvn::g_windows.create(window);
    w.native_window = b->native_window;
    w.present_interval = b->present_interval;
    NVN_LOG("ventana con intervalo de presentación %d\n", w.present_interval);
    for (int i = 0; i < b->texture_count; ++i) w.textures.push_back(nvn::g_textures.reference(b->textures[i]));
    std::lock_guard lock(g_present_mutex);
    if (create_swapchain(w)) return 1;
    nvn::g_windows.destroy(window);
    return 0;
}
void nvnWindowFinalize(void* window) {
    auto w = nvn::g_windows.take(window);
    if (!w) return;
    nvn::wait_idle();
    g_presenter_thread.wait_idle();
    std::lock_guard lock(g_present_mutex);
    {
        std::lock_guard queue_lock(nvn::vk_queue_mutex());
        vkQueueWaitIdle(nvn::vk()->queue);
    }
    destroy_swapchain(*w);
}
int nvnWindowGetPresentInterval(const void* window) {
    nvn::Window* w = nvn::g_windows.get(window);
    return w ? w->present_interval : 1;
}
void nvnWindowSetPresentInterval(void* window, int interval) {
    static int last = -1;
    if (interval != last) NVN_LOG("intervalo de presentación %d\n", interval), last = interval;
    if (nvn::Window* w = nvn::g_windows.get(window)) w->present_interval = interval;
}
int nvnWindowGetNumTextures(const void* window) {
    nvn::Window* w = nvn::g_windows.get(window);
    return w ? static_cast<int>(w->textures.size()) : 0;
}
void nvnWindowSetCrop(void* window, int x, int y, int w, int h) {
    nvn::Window* win = nvn::g_windows.get(window);
    if (!win) return;
    static VkRect2D logged{};  // DEPURACIÓN: cada recorte nuevo
    const VkRect2D crop{{x, y}, {static_cast<uint32_t>(std::max(w, 0)), static_cast<uint32_t>(std::max(h, 0))}};
    if (memcmp(&crop, &logged, sizeof crop) != 0) NVN_LOG("recorte de la ventana %d,%d %dx%d\n", x, y, w, h), logged = crop;
    win->crop = crop;
}
// NVNrectangle: x, y, ancho, alto.
void nvnWindowGetCrop(const void* window, int* rect) {
    const nvn::Window* w = nvn::g_windows.get(window);
    rect[0] = w ? w->crop.offset.x : 0, rect[1] = w ? w->crop.offset.y : 0;
    rect[2] = w ? static_cast<int>(w->crop.extent.width) : 0, rect[3] = w ? static_cast<int>(w->crop.extent.height) : 0;
}
void nvnWindowSetDebugLabel(void*, const char*) {}

}  // extern "C"

NVN_PROC(nvnWindowBuilderSetDefaults);
NVN_PROC(nvnWindowBuilderSetDevice);
NVN_PROC(nvnWindowBuilderSetNativeWindow);
NVN_PROC(nvnWindowBuilderSetPresentInterval);
NVN_PROC(nvnWindowBuilderSetTextures);
NVN_PROC(nvnWindowInitialize);
NVN_PROC(nvnWindowFinalize);
NVN_PROC(nvnWindowGetPresentInterval);
NVN_PROC(nvnWindowSetPresentInterval);
NVN_PROC(nvnWindowGetNumTextures);
NVN_PROC(nvnWindowSetCrop);
NVN_PROC(nvnWindowGetCrop);
NVN_PROC(nvnWindowSetDebugLabel);
