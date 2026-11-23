// nvnBuffer*: una ventana (offset, tamaño) sobre un memory pool. Dirección de GPU = de CPU.

#include <cstring>

#include "nnsdk/nvn/nvn.h"
#include "nnsdk/nvn/objects.h"
#include "nnsdk/nvn/commands.h"

namespace {

// NVNbufferBuilder es opaco; guardamos aquí dentro solo estos campos.
struct BufferBuilder {
    void* device;
    void* pool;
    int64_t offset;
    size_t size;
};

struct Buffer {
    uint8_t* data = nullptr;
    size_t size = 0;
    void* pool = nullptr;
    int64_t offset = 0;
};

nvn::Objects<Buffer> g_buffers;

void release_buffer(const Buffer& b) {
    auto* p = nvn::g_memory_pools.get(b.pool);
    if (p && p->reserved && b.size) p->unregister_buffer(b.offset, b.size);
}

}  // namespace

extern "C" {

void nvnBufferBuilderSetDefaults(BufferBuilder* b) { memset(b, 0, sizeof *b); }
void nvnBufferBuilderSetDevice(BufferBuilder* b, void* device) { b->device = device; }
void nvnBufferBuilderSetStorage(BufferBuilder* b, void* pool, int64_t offset, size_t size) {
    b->pool = pool;
    b->offset = offset;
    b->size = size;
}

nvn::Boolean nvnBufferInitialize(void* buffer, const BufferBuilder* b) {
    nvn::MemoryPool* pool = nvn::g_memory_pools.get(b->pool);
    if (!pool) {
        NVN_LOG("buffer %p sobre un memory pool desconocido %p\n", buffer, b->pool);
        return 0;
    }
    if (b->offset < 0 || size_t(b->offset) > pool->size || b->size > pool->size - size_t(b->offset)) return 0;
    if (auto previous = g_buffers.take(buffer)) release_buffer(*previous);
    if (pool->reserved && b->size) {
        pool->register_buffer(b->offset, b->size);
        if (!nvn::publish_virtual_buffers(b->pool)) {
            pool->unregister_buffer(b->offset, b->size);
            return 0;
        }
    }
    Buffer& buf = g_buffers.create(buffer);
    buf.data = pool->storage + b->offset;
    buf.size = b->size;
    buf.pool = b->pool;
    buf.offset = b->offset;
    return 1;
}
void nvnBufferFinalize(void* buffer) {
    if (auto b = g_buffers.take(buffer)) release_buffer(*b);
}

void* nvnBufferMap(const void* buffer) {
    Buffer* b = g_buffers.get(buffer);
    if (b) {
        const auto* p = nvn::g_memory_pools.get(b->pool);
        if (p && p->reserved && !nvn::publish_virtual_buffers(b->pool)) return nullptr;
        if (p && (p->flags & nvn::kPoolFlagPhysical) && !nvn::publish_virtual_pool(nullptr)) return nullptr;
    }
    return b ? b->data : nullptr;
}
uint64_t nvnBufferGetAddress(const void* buffer) {
    const auto* b = g_buffers.get(buffer);
    return b ? reinterpret_cast<uint64_t>(b->data) : 0;
}
size_t nvnBufferGetSize(const void* buffer) {
    Buffer* b = g_buffers.get(buffer);
    return b ? b->size : 0;
}
void* nvnBufferBuilderGetDevice(const BufferBuilder* b) { return b->device; }
void* nvnBufferBuilderGetMemoryPool(const BufferBuilder* b) { return b->pool; }
int64_t nvnBufferBuilderGetMemoryOffset(const BufferBuilder* b) { return b->offset; }
size_t nvnBufferBuilderGetSize(const BufferBuilder* b) { return b->size; }
void* nvnBufferGetMemoryPool(const void* buffer) {
    Buffer* b = g_buffers.get(buffer);
    return b ? b->pool : nullptr;
}
int64_t nvnBufferGetMemoryOffset(const void* buffer) {
    Buffer* b = g_buffers.get(buffer);
    return b ? b->offset : 0;
}
void nvnBufferSetDebugLabel(void*, const char*) {}
void nvnBufferFlushMappedRange(const void*, int64_t, size_t) {}
void nvnBufferInvalidateMappedRange(const void* buffer, int64_t, size_t size) {
    if (const auto* b = g_buffers.get(buffer); b && size) {
        const auto* p = nvn::g_memory_pools.get(b->pool);
        if (p && (p->reserved || (p->flags & nvn::kPoolFlagPhysical))) nvn::publish_virtual_pool(nullptr);
    }
}

}  // extern "C"

NVN_PROC(nvnBufferBuilderSetDefaults);
NVN_PROC(nvnBufferBuilderSetDevice);
NVN_PROC(nvnBufferBuilderSetStorage);
NVN_PROC(nvnBufferInitialize);
NVN_PROC(nvnBufferFinalize);
NVN_PROC(nvnBufferMap);
NVN_PROC(nvnBufferGetAddress);
NVN_PROC(nvnBufferGetSize);
NVN_PROC(nvnBufferBuilderGetDevice);
NVN_PROC(nvnBufferBuilderGetMemoryPool);
NVN_PROC(nvnBufferBuilderGetMemoryOffset);
NVN_PROC(nvnBufferBuilderGetSize);
NVN_PROC(nvnBufferGetMemoryPool);
NVN_PROC(nvnBufferGetMemoryOffset);
NVN_PROC(nvnBufferSetDebugLabel);
NVN_PROC(nvnBufferFlushMappedRange);
NVN_PROC(nvnBufferInvalidateMappedRange);
