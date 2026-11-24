// Memory pools the GPU reads in place. Without VK_EXT_external_memory_host the GPU can't use the game's own pages, so
// what draws read from the game's pools is copied into staging every submit, or mirrored and checked for changes (the
// "vértices" and "overlays y espejos" parts of the per-draw profile). Here a pool
// gets an Android hardware buffer (a BLOB) as big as itself, imported into Vulkan as a buffer, and that same memory
// is mapped over the pool's pages at the same address: what the game writes is what the GPU reads.
//
// The way of doing it is nxnative's (seedds/nxnative, MIT: its "zero-copy pools", gfx_vk_core.c), written again here
// for this port's pools. Only pools the game declares CPU_UNCACHED qualify: memory it fills and the GPU reads (uniform
// data, vertices, indices), which the game doesn't read back. On some phones the CPU reads a hardware buffer slowly
// (nxnative measured an Adreno 650 at about a tenth of normal memory's speed), and the renderer still reads some of
// these pools itself (copies, quads, the vertex cache's checks): there the first pool measures it and nothing is
// shared. Not the pools of
// textures (COMPRESSIBLE, PHYSICAL and VIRTUAL: the virtual ones see their physical pools' heap pages, memory.cpp), nor
// shader code. Anything that fails stays as it was: the pool's pages are untouched until the swap at the end.
// <datos>/zero_copy (at start-up, off by default).

#include "nnsdk/nvn/zero_copy.h"

#include <android/hardware_buffer.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <atomic>
#include <map>
#include <shared_mutex>

#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/nvn.h"

namespace nnsdk {
bool alias_heap_pages(void* dst, const void* src, size_t size);  // os_memory.cpp
}

namespace {

// NVNmemoryPoolFlags
constexpr int kCpuUncached = 0x2, kCpuCached = 0x4, kShaderCode = 0x40, kCompressible = 0x80, kPhysical = 0x100,
              kVirtual = 0x200;
// Some grallocs refuse a BLOB of 256 MB (nxnative: an OPPO Pad on Android 15); MH Rise's buffer pools are far smaller.
constexpr size_t kMinSize = 64 << 10, kMaxSize = (256 << 20) - 4096;

struct SharedPool {
    size_t size = 0;
    AHardwareBuffer* ahb = nullptr;
    int fd = -1;  // its dma-buf
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    bool copied = false;  // the executor wrote it mid-submit (cpu_writes): read through copies again
};

std::shared_mutex g_mutex;
std::map<uintptr_t, SharedPool> g_pools;  // by address
size_t g_bytes = 0;
std::atomic<int> g_count{0};

bool enabled() {
    static const bool on = NVN_FLAG("zero_copy");
    return on && nvn::vk() && nvn::vk()->ahb && vkGetAndroidHardwareBufferPropertiesANDROID;
}

void drop(SharedPool& p) {
    nvn::VkContext* c = nvn::vk();
    if (p.buffer) vkDestroyBuffer(c->device, p.buffer, nullptr);
    if (p.memory) vkFreeMemory(c->device, p.memory, nullptr);
    if (p.ahb) AHardwareBuffer_release(p.ahb);
    if (p.fd >= 0) close(p.fd);
    p = {};
}

// The hardware buffer's dma-buf, for mapping it. The NDK has no getter for it (AHardwareBuffer_getNativeHandle is
// the platform's), but it does send a buffer through a socket: the first fd that arrives is the dma-buf.
int buffer_fd(AHardwareBuffer* ahb) {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, sv) != 0) return -1;
    int fd = -1;
    if (AHardwareBuffer_sendHandleToUnixSocket(ahb, sv[0]) == 0) {
        char data[256];
        alignas(cmsghdr) char control[CMSG_SPACE(16 * sizeof(int))];
        iovec io{data, sizeof data};
        msghdr m{};
        m.msg_iov = &io;
        m.msg_iovlen = 1;
        m.msg_control = control;
        m.msg_controllen = sizeof control;
        if (recvmsg(sv[1], &m, MSG_CMSG_CLOEXEC) >= 0)
            for (cmsghdr* c = CMSG_FIRSTHDR(&m); c; c = CMSG_NXTHDR(&m, c)) {
                if (c->cmsg_level != SOL_SOCKET || c->cmsg_type != SCM_RIGHTS) continue;
                const int n = static_cast<int>((c->cmsg_len - CMSG_LEN(0)) / sizeof(int));
                const int* fds = reinterpret_cast<const int*>(CMSG_DATA(c));
                for (int k = 0; k < n; k++)
                    if (fd < 0) fd = fds[k];
                    else close(fds[k]);
            }
    }
    close(sv[0]);
    close(sv[1]);
    return fd;
}

uint64_t read_ns(const uint8_t* p, size_t n) {
    uint64_t best = UINT64_MAX;
    static volatile uint64_t sink;
    for (int k = 0; k < 3; k++) {
        timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        uint64_t sum = 0;
        for (size_t i = 0; i < n / 8; i++) sum += reinterpret_cast<const uint64_t*>(p)[i];
        clock_gettime(CLOCK_MONOTONIC, &t1);
        sink = sink + sum;
        best = std::min<uint64_t>(best, (t1.tv_sec - t0.tv_sec) * 1000000000ull + (t1.tv_nsec - t0.tv_nsec));
    }
    return best ? best : 1;
}

// How fast the CPU reads a hardware buffer against the heap, once (best of 3 over 1 MB each, as nxnative does).
// false where it's over 4 times slower: uncached for the CPU, not worth it.
int g_fast = -1;
bool reads_fast(const uint8_t* shared, const uint8_t* heap, size_t n) {
    int& fast = g_fast;
    if (fast >= 0) return fast;
    n = std::min<size_t>(n, 1 << 20);
    const uint64_t a = read_ns(shared, n), b = read_ns(heap, n);
    fast = a <= 4 * b;
    NVN_LOG("zero_copy: the CPU reads a hardware buffer at %.0f MB/s, the heap at %.0f MB/s%s\n", n / (a / 1e9) / 1e6,
            n / (b / 1e9) / 1e6, fast ? "" : ": uncached here, nothing is shared");
    return fast;
}

bool make(SharedPool& p, size_t size, const char** why) {
    nvn::VkContext* c = nvn::vk();
    AHardwareBuffer_Desc d{};
    d.width = static_cast<uint32_t>(size);
    d.height = 1;
    d.layers = 1;
    d.format = AHARDWAREBUFFER_FORMAT_BLOB;
    d.usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN |
              AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER;
    if (AHardwareBuffer_allocate(&d, &p.ahb) != 0) return *why = "AHardwareBuffer_allocate", false;
    p.size = size;

    VkAndroidHardwareBufferPropertiesANDROID props{VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID};
    if (vkGetAndroidHardwareBufferPropertiesANDROID(c->device, p.ahb, &props) != VK_SUCCESS || props.allocationSize < size)
        return *why = "vkGetAndroidHardwareBufferPropertiesANDROID", false;

    VkExternalMemoryBufferCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &ext};
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(c->device, &bi, nullptr, &p.buffer) != VK_SUCCESS) return *why = "vkCreateBuffer", false;

    uint32_t type = UINT32_MAX;
    for (uint32_t k = 0; k < c->memory.memoryTypeCount && type == UINT32_MAX; k++)
        if (props.memoryTypeBits >> k & 1) type = k;
    if (type == UINT32_MAX) return *why = "no memory type", false;
    VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.buffer = p.buffer;
    VkImportAndroidHardwareBufferInfoANDROID import{VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID, &dedicated};
    import.buffer = p.ahb;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &import};
    ai.allocationSize = props.allocationSize;
    ai.memoryTypeIndex = type;
    if (vkAllocateMemory(c->device, &ai, nullptr, &p.memory) != VK_SUCCESS) return *why = "vkAllocateMemory", false;
    if (vkBindBufferMemory(c->device, p.buffer, p.memory, 0) != VK_SUCCESS) return *why = "vkBindBufferMemory", false;
    return true;
}

}  // namespace

bool nvn::zero_copy::adopt(void* storage, size_t size, int flags) {
    if (!enabled() || !storage || g_fast == 0) return false;
    const uintptr_t at = reinterpret_cast<uintptr_t>(storage);
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    if (!(flags & kCpuUncached) || (flags & (kCpuCached | kShaderCode | kCompressible | kPhysical | kVirtual)) ||
        (at | size) & (page - 1) || size < kMinSize || size > kMaxSize) {
        static int skipped = 0;  // (which pools the game makes, for choosing better)
        if (skipped++ < 32) NVN_LOG("zero_copy: pool %p of %zu KB (flags 0x%x) not shared\n", storage, size >> 10, flags);
        return false;
    }

    // The way back (release, or a failed swap) is the heap's own pages at the same place: only where those are a memfd
    // (os_memory.cpp). Mapping them over themselves changes nothing.
    if (!nnsdk::alias_heap_pages(storage, storage, size)) return false;
    static int logged = 0;
    SharedPool p;
    const char* why = nullptr;
    bool ok = make(p, size, &why);
    // The game's data into the hardware buffer, then the buffer over the pool's pages
    if (ok && (p.fd = buffer_fd(p.ahb)) < 0) ok = false, why = "no dma-buf";
    if (ok) {
        void* tmp = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, p.fd, 0);
        if (tmp == MAP_FAILED) {
            ok = false, why = "mmap";
        } else {
            memcpy(tmp, storage, size);
            const bool fast = reads_fast(static_cast<const uint8_t*>(tmp), static_cast<const uint8_t*>(storage), size);
            munmap(tmp, size);
            if (!fast) {
                ok = false, why = "the read check";
            } else if (mmap(storage, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, p.fd, 0) != storage) {
                // (the old mapping may be gone: the heap's own pages again, with the data, are the safe way back)
                nnsdk::alias_heap_pages(storage, storage, size);
                ok = false, why = "mmap over the pool";
            }
        }
    }
    if (!ok) {
        if (logged++ < 8) NVN_LOG("zero_copy: pool %p of %zu KB (flags 0x%x) stays copied: %s failed\n", storage, size >> 10, flags, why);
        drop(p);
        return false;
    }
    std::unique_lock lock(g_mutex);
    g_pools[at] = p;
    g_bytes += size;
    g_count.fetch_add(1, std::memory_order_relaxed);
    if (logged++ < 24)
        NVN_LOG("zero_copy: pool %p of %zu KB (flags 0x%x) is shared with the GPU (%zu MB in all)\n", storage, size >> 10,
                flags, g_bytes >> 20);
    return true;
}

void nvn::zero_copy::release(void* storage) {
    SharedPool p;
    {
        std::unique_lock lock(g_mutex);
        auto it = g_pools.find(reinterpret_cast<uintptr_t>(storage));
        if (it == g_pools.end()) return;
        p = it->second;
        g_bytes -= p.size;
        g_pools.erase(it);
        g_count.fetch_sub(1, std::memory_order_relaxed);
    }
    // What the pool holds now, back on the heap's pages (the game may go on using the memory)
    void* tmp = mmap(nullptr, p.size, PROT_READ, MAP_SHARED, p.fd, 0);
    nnsdk::alias_heap_pages(storage, storage, p.size);
    if (tmp != MAP_FAILED) {
        memcpy(storage, tmp, p.size);
        munmap(tmp, p.size);
    }
    drop(p);
}

bool nvn::zero_copy::any() { return g_count.load(std::memory_order_relaxed) > 0; }

void nvn::zero_copy::cpu_writes(uint64_t address, uint64_t size) {
    if (!any() || !size) return;
    std::unique_lock lock(g_mutex);
    auto it = g_pools.upper_bound(address + size - 1);
    while (it != g_pools.begin()) {
        --it;
        if (it->first + it->second.size <= address) break;
        if (it->second.copied) continue;
        it->second.copied = true;
        static int logged = 0;
        if (logged++ < 16)
            NVN_LOG("zero_copy: pool %#llx of %zu KB is written mid-submit (%#llx, %llu bytes): copied again\n",
                    static_cast<unsigned long long>(it->first), it->second.size >> 10, static_cast<unsigned long long>(address),
                    static_cast<unsigned long long>(size));
    }
}

bool nvn::zero_copy::find(uint64_t address, uint64_t size, VkBuffer* buffer, VkDeviceSize* offset) {
    if (!any()) return false;
    std::shared_lock lock(g_mutex);
    auto it = g_pools.upper_bound(address);
    if (it == g_pools.begin()) return false;
    --it;
    const uint64_t at = address - it->first;
    if (at >= it->second.size || size > it->second.size - at || (buffer && it->second.copied)) return false;
    if (buffer) *buffer = it->second.buffer;
    if (offset) *offset = at;
    return true;
}
