#include "nnsdk/nvn/driver_pipeline_cache.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/nvn.h"
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <sys/resource.h>
#include <thread>
#include <unistd.h>

namespace nvn {

namespace {
std::string pipeline_cache_path() {
    const char* data = getenv("NIER_DATA");
    return std::string(data ? data : "/sdcard/NieR") + "/pipelines.bin";
}

// data en path, por path.tmp y rename (con fsync antes: un corte deja el fichero de antes entero), a ritmo: trozos de 1
// MB con una pausa entre ellos y fdatasync cada 4 MB. De un tirón, los ~25 MB de pipelines.bin llenaban la cola de la SD
// y lo demás que la toca (los paks que lee el juego, el log) esperaba detrás; así nunca hay más de unos MB nuestros por
// delante. Tarda algo más (las pausas: ~0,2 s en 25 MB), pero en un hilo que nadie espera.
bool write_file_paced(const std::string& path, const uint8_t* data, size_t size) {
    constexpr size_t kChunk = 1u << 20, kSyncEvery = 4u << 20;
    constexpr auto kPause = std::chrono::milliseconds(8);
    const std::string tmp = path + ".tmp";
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
    if (fd < 0) return false;
    bool ok = true;
    size_t done = 0, unsynced = 0;
    while (ok && done < size) {
        const size_t len = std::min(kChunk, size - done);
        for (size_t put = 0; ok && put < len;) {
            const ssize_t w = ::write(fd, data + done + put, len - put);
            if (w < 0 && errno == EINTR) continue;
            ok = w > 0;
            if (ok) put += static_cast<size_t>(w);
        }
        done += len;
        unsynced += len;
        if (ok && unsynced >= kSyncEvery) {
            fdatasync(fd);
            unsynced = 0;
        }
        if (ok && done < size) std::this_thread::sleep_for(kPause);
    }
    ok = ok && fsync(fd) == 0;
    ok = close(fd) == 0 && ok;
    ok = ok && rename(tmp.c_str(), path.c_str()) == 0;
    if (!ok) unlink(tmp.c_str());
    return ok;
}
}  // namespace

// La cabecera se comprueba antes de dársela al driver (un fichero de otro driver o roto se descarta).
void DriverPipelineCache::load() {
    VkContext* c = vk();
    std::vector<uint8_t> data;
    if (FILE* f = fopen(pipeline_cache_path().c_str(), "rb")) {
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (size > 32 && size < (128l << 20)) {
            data.resize(static_cast<size_t>(size));
            if (fread(data.data(), 1, data.size(), f) != data.size()) data.clear();
        }
        fclose(f);
    }
    if (!data.empty()) {
        uint32_t header[4];
        memcpy(header, data.data(), sizeof header);
        const bool ok = header[0] >= 32 && header[1] == VK_PIPELINE_CACHE_HEADER_VERSION_ONE &&
                        header[2] == c->props.vendorID && header[3] == c->props.deviceID &&
                        memcmp(data.data() + 16, c->props.pipelineCacheUUID, VK_UUID_SIZE) == 0;
        if (!ok) data.clear();
    }
    VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    ci.initialDataSize = data.size();
    ci.pInitialData = data.empty() ? nullptr : data.data();
    if (vkCreatePipelineCache(c->device, &ci, nullptr, &pipeline_cache_) != VK_SUCCESS) {
        data.clear();
        ci.initialDataSize = 0;
        ci.pInitialData = nullptr;
        if (vkCreatePipelineCache(c->device, &ci, nullptr, &pipeline_cache_) != VK_SUCCESS) pipeline_cache_ = VK_NULL_HANDLE;
    }
    NVN_LOG("caché de pipelines: %zu KB leídos de la SD\n", data.size() >> 10);
    if (!NVN_FLAG("no_warmup")) warm_cache_data_ = std::move(data);  // la caché del precalentado empieza igual
}

// En otro hilo (escribir en la SD es lento) y solo cuando lleva un rato sin crearse ninguno. La caché la
// sincroniza el driver: se puede leer mientras el ejecutor crea pipelines (Vulkan solo pide sincronizarla por fuera al
// mezclar en ella y al destruirla: pipeline_cache_lock_, exclusivo para la mezcla del precalentado).
// Cada guardado reescribe la caché entera (21 MB en MH Rise). Antes bastaban 2 s sin pipelines nuevas: jugando se
// guardaba hasta 10 veces por minuto (59 en 16 min del 4-11, 1,2 GB a la SD). A ratos la SD no daba abasto (un guardado
// tardaba 4-7 s en vez de 0,6) y mientras todo lo que la toca esperaba (el log, los interruptores, los paks): de 30 a
// 1-3 fps con el juego quieto, con 0 pipelines y 0 traducciones nuevas en esos frames. Ahora: 4 s sin pipelines nuevas
// y como mucho un guardado cada 3 min (cada 30 s si hay 300 nuevas: tras cambiar los shaders salen miles).
// <datos>/fast_pipeline_save: como antes.
// El hilo: nunca dos a la vez (cache_saving_). Lee los datos del driver con prioridad normal (con el cerrojo compartido:
// con menos, la mezcla del precalentado podría quedarse esperando a un hilo que no corre, y el ejecutor detrás de ella)
// y escribe el fichero con la prioridad más baja y a ritmo (write_file_paced). El log da cada parte por
// separado: "driver" (lo que el ejecutor podría esperar dentro del driver si crea una pipeline a la vez: lo dice
// note_pipeline_during_save) y "SD".
void DriverPipelineCache::save(uint64_t frame) {
    if (const uint64_t warm = warm_new_.exchange(0)) {  // mezcladas desde la caché del precalentado
        pipelines_created_ += warm;
        last_pipeline_frame_ = frame;
    }
    if (!pipeline_cache_ || pipelines_created_ == pipelines_saved_) return;
    const bool quick = NVN_FLAG("fast_pipeline_save");
    if (frame < last_pipeline_frame_ + (quick ? 60 : 120)) return;
    const auto now = std::chrono::steady_clock::now();
    const uint64_t fresh = pipelines_created_ - pipelines_saved_;
    if (!quick && now - pipelines_saved_at_ < std::chrono::seconds(fresh >= 300 ? 30 : 180)) return;
    if (cache_saving_.exchange(true)) return;
    const uint64_t snapshot = pipelines_created_;
    pipelines_saved_at_ = now;
    const VkDevice device = vk()->device;
    const VkPipelineCache cache = pipeline_cache_;
    std::thread([this, device, cache, fresh, snapshot] {
        pthread_setname_np(pthread_self(), "GuardaPipelines");
        // Lo crea NvnSubmit (presenta y llama a profile_frame): hereda su nice -10 (kSubmitNice) y su máscara de núcleos
        // grandes (esta la devuelve os.cpp en la revisión de cada segundo). Para leer la caché, como un hilo del juego.
        setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), 0);
        const auto start = std::chrono::steady_clock::now();
        std::vector<uint8_t> data;
        {
            std::shared_lock cache_lock(pipeline_cache_lock_);  // no mientras el precalentado mezcla la suya
            cache_reading_ = true;
            size_t size = 0;
            if (vkGetPipelineCacheData(device, cache, &size, nullptr) == VK_SUCCESS && size) {
                data.resize(size);
                if (vkGetPipelineCacheData(device, cache, &size, data.data()) != VK_SUCCESS) data.clear();
                data.resize(std::min(size, data.size()));
            }
            cache_reading_ = false;
        }
        const auto got = std::chrono::steady_clock::now();
        setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), 19);  // ya sin el cerrojo: por debajo de todo
        const bool written = !data.empty() && write_file_paced(pipeline_cache_path(), data.data(), data.size());
        const auto ms = [](std::chrono::steady_clock::duration d) {
            return static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(d).count());
        };
        // Para medir: lo que tarda cada guardado (0,5-0,9 s con la SD sana) y cuántos hay.
        NVN_LOG("caché de pipelines %s: %zu KB (%llu nuevas) en %lld ms (driver %lld ms, SD %lld ms, en segundo plano)\n",
                written ? "guardada" : "SIN guardar", data.size() >> 10, static_cast<unsigned long long>(fresh),
                ms(std::chrono::steady_clock::now() - start), ms(got - start), ms(std::chrono::steady_clock::now() - got));
        if (written) pipelines_saved_ = snapshot;
        cache_saving_ = false;
    }).detach();
}

void DriverPipelineCache::merge(VkPipelineCache source, uint64_t created) {
    if (!created || !pipeline_cache_) return;
    std::unique_lock lock(pipeline_cache_lock_);
    if (vkMergePipelineCaches(vk()->device, pipeline_cache_, 1, &source) == VK_SUCCESS) warm_new_ += created;
}

}  // namespace nvn
