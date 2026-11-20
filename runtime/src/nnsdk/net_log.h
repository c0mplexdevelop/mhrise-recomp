// Local Play capture (mhr.ini net_log, Settings → Debug → Capture Local Play): every datagram the game sends into the
// network or gets from it, written to <data>/netlog/local_play_<date>.txt: a line with the time, sender, receiver and
// size, then the bytes in hex with their ASCII beside them. What goes over the network is Pia's packets (the game's
// session layer on the Switch), so this is what a bridge to another platform's multiplayer would have to speak.
// Turned on and off live (checked every 2 s); a new file each time it's switched on.
#pragma once

#include <sys/stat.h>
#include <sys/time.h>

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <string>

#include "nnsdk/ini.h"
#include "nnsdk/ldn_net.h"

namespace nnsdk::net_log {

struct State {
    std::mutex m;
    FILE* f = nullptr;
    bool on = false;
    long long checked_ms = -10000;
    unsigned long long packets = 0;
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
                printf("[net_log] capturing Local Play to netlog%s\n", name);
            }
        } else if (!on && s.f) {
            fclose(s.f);
            s.f = nullptr;
            printf("[net_log] capture closed: %llu datagrams\n", s.packets);
        }
        if (s.f) fflush(s.f);
    }
    if (!s.f) return;
    const time_t sec = static_cast<time_t>(ms / 1000);
    tm lt;
    localtime_r(&sec, &lt);
    const bool out = src == ldn_net::own_address();
    fprintf(s.f, "\n%02d:%02d:%02d.%03d %c %u.%u.%u.%u:%u -> %u.%u.%u.%u:%u  %zu bytes\n", lt.tm_hour, lt.tm_min, lt.tm_sec,
            static_cast<int>(ms % 1000), out ? '>' : '<', src >> 24, (src >> 16) & 255, (src >> 8) & 255, src & 255,
            src_port, dst >> 24, (dst >> 16) & 255, (dst >> 8) & 255, dst & 255, dst_port, size);
    const auto* b = static_cast<const uint8_t*>(data);
    for (size_t row = 0; row < size; row += 16) {
        char line[96];
        int n = snprintf(line, sizeof line, "  %04zx ", row);
        for (size_t k = 0; k < 16; k++)
            n += row + k < size ? snprintf(line + n, sizeof line - n, " %02x", b[row + k])
                                : snprintf(line + n, sizeof line - n, "   ");
        n += snprintf(line + n, sizeof line - n, "  ");
        for (size_t k = 0; k < 16 && row + k < size; k++) line[n++] = b[row + k] >= 32 && b[row + k] < 127 ? b[row + k] : '.';
        line[n] = 0;
        fprintf(s.f, "%s\n", line);
    }
    s.packets++;
}

}  // namespace nnsdk::net_log
