#pragma once
// The key Pia encrypts a Local Play network's packets with (AES-128-GCM), for a PC player's bridge in it (ldn.cpp:
// a station whose hello says it's a bridge gets it in a KEY frame, then again whenever it changes). Pia makes a new one
// for each network and expands it with its key schedule (main+0x24dab50): on the host from main+0x24d9ae4 when the
// network is made, on a station from main+0x24d9cb0 when it joins. The newest from either is the network's (checked
// against captures: every packet of three sessions verifies with it; the one 0x24d9ae4 expands on the title screen,
// before any network, is replaced by then). MH Rise 16.0.1 only: the code's bytes are checked before anything is
// patched.
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace nnsdk::pia_key {

inline const uint8_t* g_main = nullptr;
inline uint64_t (*g_schedule)(void*, const uint8_t*, uint32_t, uint32_t) = nullptr;  // the original, moved
inline std::mutex g_mutex;
inline uint8_t g_key[16] = {};
inline std::atomic<uint32_t> g_serial{0};  // 0: none yet
inline bool g_keep = true;                  // keep the network's key (off with no_pc_players)
// Told about every key Pia expands, with its caller's offset in main (the Local Play capture, net_log.h)
inline void (*g_observer)(const uint8_t* key, uint32_t len, uint32_t mode, uintptr_t caller) = nullptr;

// The network's key now, and a number that changes with it (0: none yet)
inline uint32_t current(uint8_t out[16]) {
    std::lock_guard lock(g_mutex);
    memcpy(out, g_key, 16);
    return g_serial.load();
}

extern "C" inline uint64_t pia_key_schedule(void* ctx, const uint8_t* key, uint32_t len, uint32_t mode) {
    const auto caller = reinterpret_cast<uintptr_t>(__builtin_return_address(0)) - reinterpret_cast<uintptr_t>(g_main);
    if (g_observer && key && len && len <= 32) g_observer(key, len, mode, caller);
    if (g_keep && key && len == 16 && (caller == 0x24d9ae4 || caller == 0x24d9cb0)) {
        std::lock_guard lock(g_mutex);
        if (memcmp(g_key, key, 16) != 0) {
            memcpy(g_key, key, 16);
            g_serial++;
        }
    }
    return g_schedule(ctx, key, len, mode);
}

// main+0x24dab50 starts str x22, [sp, #-0x30]!; stp x21, x20, [sp, #0x10]; stp x19, x30, [sp, #0x20]; mov w22, w2:
// nothing that depends on where it runs, so the four go to a trampoline that jumps back after them, and their place
// gets a jump to pia_key_schedule.
inline void hook(const uint8_t* main_base, bool keep = true) {
    g_keep = keep;
    static const uint32_t kOriginal[4] = {0xf81d0ff6, 0xa90153f5, 0xa9027bf3, 0x2a0203f6};
    auto* target = const_cast<uint8_t*>(main_base) + 0x24dab50;
    if (memcmp(target, kOriginal, sizeof kOriginal) != 0) {
        printf("[ldn] Pia's key schedule isn't where 16.0.1 has it: no PC players, no keys in a capture\n");
        return;
    }
    auto* tramp = static_cast<uint32_t*>(mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (tramp == MAP_FAILED) return;
    memcpy(tramp, kOriginal, sizeof kOriginal);
    tramp[4] = 0x58000050;  // ldr x16, #8
    tramp[5] = 0xd61f0200;  // br x16
    const uint64_t back = reinterpret_cast<uint64_t>(target + 16);
    memcpy(tramp + 6, &back, 8);
    mprotect(tramp, 4096, PROT_READ | PROT_EXEC);
    __builtin___clear_cache(reinterpret_cast<char*>(tramp), reinterpret_cast<char*>(tramp + 8));
    g_main = main_base;
    g_schedule = reinterpret_cast<decltype(g_schedule)>(tramp);

    uint32_t jump[4] = {0x58000050, 0xd61f0200};
    const uint64_t to = reinterpret_cast<uint64_t>(&pia_key_schedule);
    memcpy(jump + 2, &to, 8);
    const uint64_t page = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
    auto* first = reinterpret_cast<uint8_t*>(reinterpret_cast<uint64_t>(target) & ~(page - 1));
    const size_t span = static_cast<size_t>(target + 16 - first + page - 1) & ~(page - 1);
    if (mprotect(first, span, PROT_READ | PROT_WRITE | PROT_EXEC) != 0) {
        printf("[ldn] can't patch main's code: no PC players\n");
        return;
    }
    memcpy(target, jump, sizeof jump);
    mprotect(first, span, PROT_READ | PROT_EXEC);
    __builtin___clear_cache(reinterpret_cast<char*>(target), reinterpret_cast<char*>(target + 16));
}

}  // namespace nnsdk::pia_key
