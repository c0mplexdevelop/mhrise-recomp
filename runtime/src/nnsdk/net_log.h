// Local Play capture (mhr.ini net_log, Settings → Debug → Capture Local Play): every datagram the game sends into the
// network or gets from it, written to <data>/netlog/local_play_<date>.txt: a line with the time, sender, receiver and
// size, then the bytes in hex with their ASCII beside them. What goes over the network is Pia's packets (the game's
// session layer on the Switch), so this is what a bridge to another platform's multiplayer would have to speak.
// Turned on and off live (checked every 2 s); a new file each time it's switched on.
//
// Pia encrypts them (AES-GCM, the 0x80 in the version byte), so the file also gets what's needed to take that off
// afterwards: the network's session id, its players' MACs and advertise data (the nonce is made from them), and every
// AES key the game expands while net_log is on at start-up (note_key, from pia_key.h's hook on Pia's key schedule).
#pragma once

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <vector>

#include "nnsdk/ini.h"
#include "nnsdk/ldn_net.h"

namespace nnsdk::net_log {

struct State {
    std::mutex m;
    FILE* f = nullptr;
    bool on = false;
    long long checked_ms = -10000;
    unsigned long long packets = 0;
    std::vector<std::string> keys;  // "key hex, mode, caller" lines, all of them since start-up
    size_t keys_written = 0;        // how many of them the current file has
    std::vector<uint8_t> info;      // the NetworkInfo last written
};

inline State& state() {
    static State s;
    return s;
}

inline long long now_ms() {
    timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1000LL + tv.tv_usec / 1000;
}

inline void hex(FILE* f, const char* prefix, const uint8_t* b, size_t size) {
    for (size_t row = 0; row < size; row += 16) {
        char line[112];
        int n = snprintf(line, sizeof line, "%s%04zx ", prefix, row);
        for (size_t k = 0; k < 16; k++)
            n += row + k < size ? snprintf(line + n, sizeof line - n, " %02x", b[row + k])
                                : snprintf(line + n, sizeof line - n, "   ");
        n += snprintf(line + n, sizeof line - n, "  ");
        for (size_t k = 0; k < 16 && row + k < size; k++) line[n++] = b[row + k] >= 32 && b[row + k] < 127 ? b[row + k] : '.';
        line[n] = 0;
        fprintf(f, "%s\n", line);
    }
}

// The keys found since the file was last written, and the network if it changed (s.m held)
inline void write_context(State& s) {
    for (; s.keys_written < s.keys.size(); s.keys_written++) fprintf(s.f, "\n# AES key %s\n", s.keys[s.keys_written].c_str());
    uint8_t info[0x480];
    if (!ldn_net::network_info(info) || (s.info.size() == sizeof info && !memcmp(s.info.data(), info, sizeof info))) return;
    s.info.assign(info, info + sizeof info);
    fprintf(s.f, "\n# network: local id %016llx, session id", static_cast<unsigned long long>(
                                                                     info[0] | uint64_t{info[1]} << 8 | uint64_t{info[2]} << 16 |
                                                                     uint64_t{info[3]} << 24 | uint64_t{info[4]} << 32 |
                                                                     uint64_t{info[5]} << 40 | uint64_t{info[6]} << 48 |
                                                                     uint64_t{info[7]} << 56));
    for (int k = 0; k < 16; k++) fprintf(s.f, " %02x", info[0x10 + k]);
    fprintf(s.f, "\n# security parameter");
    for (int k = 0; k < 16; k++) fprintf(s.f, " %02x", info[0x50 + k]);
    fprintf(s.f, "\n");
    for (int n = 0; n < info[0x66] && n < 8; n++) {
        const uint8_t* node = info + 0x68 + n * 0x40;
        if (!node[0x0B]) continue;
        fprintf(s.f, "# player %d: %u.%u.%u.%u mac %02x:%02x:%02x:%02x:%02x:%02x\n", node[0x0A], node[3], node[2], node[1],
                node[0], node[4], node[5], node[6], node[7], node[8], node[9]);
    }
    const size_t adv = std::min<size_t>(info[0x26A] | info[0x26B] << 8, sizeof info - 0x26C);
    fprintf(s.f, "# advertise data (%zu bytes):\n", adv);
    hex(s.f, "#   ", info + 0x26C, adv);
}

// Addresses and ports in host byte order
inline void datagram(uint32_t src, uint16_t src_port, uint32_t dst, uint16_t dst_port, const void* data, size_t size) {
    State& s = state();
    std::lock_guard lock(s.m);
    const long long ms = now_ms();
    if (ms - s.checked_ms >= 2000) {
        s.checked_ms = ms;
        const bool on = ini::on("net_log");
        if (on && !s.f) {
            const std::string dir = ini::data_dir() + "/netlog";
            mkdir(dir.c_str(), 0775);
            char name[64];
            const time_t t = time(nullptr);
            strftime(name, sizeof name, "/local_play_%Y%m%d_%H%M%S.txt", localtime(&t));
            s.f = fopen((dir + name).c_str(), "w");
            if (s.f) {
                fprintf(s.f, "# Local Play capture: > sent by this phone, < received. Addresses are the network's.\n");
                s.packets = 0;
                s.keys_written = 0;
                s.info.clear();
                printf("[net_log] capturing Local Play to netlog%s\n", name);
            }
        } else if (!on && s.f) {
            fclose(s.f);
            s.f = nullptr;
            printf("[net_log] capture closed: %llu datagrams\n", s.packets);
        }
        if (s.f) {
            write_context(s);
            fflush(s.f);
        }
    }
    if (!s.f) return;
    const time_t sec = static_cast<time_t>(ms / 1000);
    tm lt;
    localtime_r(&sec, &lt);
    const bool out = src == ldn_net::own_address();
    fprintf(s.f, "\n%02d:%02d:%02d.%03d %c %u.%u.%u.%u:%u -> %u.%u.%u.%u:%u  %zu bytes\n", lt.tm_hour, lt.tm_min, lt.tm_sec,
            static_cast<int>(ms % 1000), out ? '>' : '<', src >> 24, (src >> 16) & 255, (src >> 8) & 255, src & 255,
            src_port, dst >> 24, (dst >> 16) & 255, (dst >> 8) & 255, dst & 255, dst_port, size);
    hex(s.f, "  ", static_cast<const uint8_t*>(data), size);
    s.packets++;
}

// ---- Pia's AES keys -------------------------------------------------------------------------------------------------

// Every key Pia expands (pia_key.h hooks its key schedule and calls this), with its caller in main
inline void note_key(const uint8_t* key, uint32_t len, uint32_t mode, uintptr_t caller) {
    char line[160];
    int n = 0;
    for (uint32_t k = 0; k < len; k++) n += snprintf(line + n, sizeof line - n, "%02x", key[k]);
    snprintf(line + n, sizeof line - n, " (%u bytes, mode %u, from main+%#lx)", len, mode, static_cast<unsigned long>(caller));
    State& s = state();
    std::lock_guard lock(s.m);
    bool seen = false;
    for (const std::string& k : s.keys) seen |= k == line;
    if (!seen && s.keys.size() < 256) {
        s.keys.emplace_back(line);
        printf("[net_log] AES key %s\n", line);
    }
}

}  // namespace nnsdk::net_log
