#include "nnsdk/nvn/vertex_fetch.h"
#include <spirv/unified1/spirv.h>
#include <cstddef>
#include <map>
#include <tuple>

namespace nvn {
namespace {
using Fetch = VertexFetch;

// SPIR-V logical addressing. Change the type at the fetch and restore the old SSA
// result at the load. All downstream arithmetic, control flow and decorations keep their IDs.
struct Type {
    uint32_t op = 0, a = 0, b = 0;
};
using TypeKey = std::tuple<uint32_t, uint32_t, uint32_t>;
struct Types {
    std::map<TypeKey, uint32_t> ids;
    std::vector<Type> original;
    std::vector<uint32_t> declarations;
    uint32_t next;

    uint32_t get(uint32_t op, uint32_t a, uint32_t b) {
        auto [it, fresh] = ids.try_emplace({op, a, b}, next);
        if (fresh) declarations.insert(declarations.end(), {(4u << 16) | op, next++, a, b});
        return it->second;
    }
    uint32_t numeric(uint32_t old, Fetch fetch) {
        if (old >= original.size()) return 0;
        const Type t = original[old];
        const bool vec = t.op == SpvOpTypeVector;
        if (vec && t.a >= original.size()) return 0;
        const Type scalar = vec ? original[t.a] : t;
        if (scalar.op != SpvOpTypeFloat || scalar.a != 32) return 0;
        const uint32_t integer = get(SpvOpTypeInt, 32, fetch == Fetch::Signed || fetch == Fetch::SignedScaled);
        return vec ? get(SpvOpTypeVector, integer, t.b) : integer;
    }
};
} // namespace

bool specialize_vertex_fetch(const std::vector<uint32_t>& w, uint64_t signature, std::vector<uint32_t>& output) {
    output.clear();
    if (!signature) return true;
    if (w.size() < 5 || w[0] != SpvMagicNumber || !w[3] || w[3] > 0x3fffff) return false;
    const uint32_t bound = w[3];
    Types types{{}, std::vector<Type>(bound), {}, bound};
    std::vector<Fetch> location(bound), pointer(bound);
    std::vector<uint32_t> globals, variables, body;
    bool vertex = false, changed = false;
    size_t first_function = w.size();
    for (size_t i = 5; i < w.size();) {
        const uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (!n || n > w.size() - i) return false;
        if (op == SpvOpFunction) {
            first_function = i;
            break;
        }
        if (op == SpvOpEntryPoint && n >= 4) vertex |= w[i + 1] == SpvExecutionModelVertex;
        if (op == SpvOpDecorate && n == 4 && w[i + 2] == SpvDecorationLocation && w[i + 3] < 16 && w[i + 1] < bound)
            location[w[i + 1]] = Fetch((signature >> (4 * w[i + 3])) & 7);  // (bit 3, w read as 1, only in the typed translation)
        if ((op == SpvOpTypeFloat && n == 3) ||
            ((op == SpvOpTypeInt || op == SpvOpTypeVector || op == SpvOpTypePointer) && n == 4)) {
            if (w[i + 1] >= bound) return false;
            Type t{op, w[i + 2], n == 4 ? w[i + 3] : 0};
            types.original[w[i + 1]] = t;
            types.ids.emplace(TypeKey{op, t.a, t.b}, w[i + 1]);
        }
        i += n;
    }
    if (!vertex || first_function == w.size()) return false;
    globals.assign(w.begin(), w.begin() + 5);
    for (size_t i = 5; i < first_function;) {
        const uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op == SpvOpVariable && n == 4 && w[i + 3] == SpvStorageClassInput && w[i + 2] < bound &&
            location[w[i + 2]] != Fetch::Float) {
            const auto fetch = location[w[i + 2]];
            if (w[i + 1] >= bound) return false;
            const auto p = types.original[w[i + 1]];
            if (p.op != SpvOpTypePointer || p.a != SpvStorageClassInput || p.b >= bound) return false;
            const Type t = types.original[p.b];
            if (t.op == SpvOpTypeVector && t.a >= bound) return false;
            const Type scalar = t.op == SpvOpTypeVector ? types.original[t.a] : t;
            // Previously translated typed inputs already perform their own conversion.
            const bool typed = scalar.op == SpvOpTypeInt && scalar.a == 32 &&
                               scalar.b == unsigned(fetch == Fetch::Signed || fetch == Fetch::SignedScaled);
            if (!typed) {
                const auto integer = types.numeric(p.b, fetch);
                if (!integer) return false;
                variables.insert(variables.end(), {w[i], types.get(SpvOpTypePointer, SpvStorageClassInput, integer),
                                                   w[i + 2], SpvStorageClassInput});
                pointer[w[i + 2]] = fetch;
                changed = true;
                i += n;
                continue;
            }
        }
        globals.insert(globals.end(), w.begin() + i, w.begin() + i + n);
        i += n;
    }
    if (!changed) return true;
    for (size_t i = first_function; i < w.size();) {
        const uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (!n || n > w.size() - i) return false;
        const size_t at = body.size();
        body.insert(body.end(), w.begin() + i, w.begin() + i + n);
        const auto kind = [&](uint32_t id) { return id < bound ? pointer[id] : Fetch::Float; };
        if ((op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain || op == SpvOpCopyObject || op == SpvOpLoad) &&
            n >= 4 && kind(w[i + 3]) != Fetch::Float) {
            const auto fetch = kind(w[i + 3]);
            const auto type = w[i + 1], result = w[i + 2];
            if (type >= bound || result >= bound) return false;
            if (op == SpvOpLoad) {
                const auto integer = types.numeric(type, fetch);
                if (!integer) return false;
                const uint32_t raw = types.next++;
                body[at + 1] = integer;
                body[at + 2] = raw;
                const uint32_t convert = fetch == Fetch::SignedScaled     ? SpvOpConvertSToF
                                         : fetch == Fetch::UnsignedScaled ? SpvOpConvertUToF
                                                                          : SpvOpBitcast;
                body.insert(body.end(), {(4u << 16) | convert, type, result, raw});
            } else {
                const auto p = types.original[type];
                if (p.op != SpvOpTypePointer || p.a != SpvStorageClassInput) return false;
                const auto integer = types.numeric(p.b, fetch);
                if (!integer) return false;
                body[at + 1] = types.get(SpvOpTypePointer, SpvStorageClassInput, integer);
                pointer[result] = fetch;
            }
        } else if (op == SpvOpFunctionCall && n >= 4) {
            for (size_t k = i + 4; k < i + n; k++)
                if (kind(w[k]) != Fetch::Float) return false;
        } else if ((op == SpvOpCopyMemory || op == SpvOpCopyMemorySized) && n >= 3 && kind(w[i + 2]) != Fetch::Float)
            return false;
        i += n;
    }
    globals[3] = types.next;
    globals.insert(globals.end(), types.declarations.begin(), types.declarations.end());
    globals.insert(globals.end(), variables.begin(), variables.end());
    globals.insert(globals.end(), body.begin(), body.end());
    output = std::move(globals);
    return true;
}
} // namespace nvn
