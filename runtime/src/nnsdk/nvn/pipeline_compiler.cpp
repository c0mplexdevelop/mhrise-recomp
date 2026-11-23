#include "nnsdk/nvn/pipeline_compiler.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/formats.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_set>

namespace nvn {
using nvnshader::Stage;

namespace {

// Pipelines that crash the driver. Some drivers kill the whole game inside vkCreateGraphicsPipelines: ARM's older ones
// (a Dimensity 720's Mali-G57) abort on some vertex shaders ("LLVM ERROR: Cannot select" in __start__position_shader)
// whose SPIR-V has nothing unusual, Qualcomm's segfaulted on a pipeline worker (an Adreno 732). There's nothing to catch
// and nothing to fix from here. So each build marks its pipeline (the key's hash: the same content gives the same one
// in every run) in a small file mapped in memory, a page the kernel keeps when the process dies, and clears it after;
// a mark still there at the next start was a build the driver died in, and that pipeline goes to the skip list
// (<data>/pipeline_skip.txt): it isn't built again and its draws are left out. So is every other pipeline with
// the same vertex shader reading the same formats: ARM's driver compiles a position-only copy of the vertex shader
// from just those two, and on the G57 the same one died again under another pass and pixel shader. One crash per such
// vertex shader, then the game goes on without that object. <data>/no_pipeline_skip ignores the list.
constexpr int kGuardSlots = 64;
// After the slots, for each build in progress: its vertex half (vertex_id) and what it reads from its vertex buffers
constexpr int kGuardNote = 192;
struct GuardNote {
    uint64_t vertex;
    char text[kGuardNote - 8];
};
// The list's first line. Bumped when a translator change can fix what crashed (2: the point size on ARM's driver; 3:
// tessellation in its domain), so an older list is dropped and those pipelines are tried again, with the crash the
// last run left marked (it was the older translation too).
constexpr const char* kSkipHeader = "pipeline_skip 3";
struct Guard {
    std::atomic<uint64_t>* slots = nullptr;  // in the mapped file: the pipeline of each build in progress
    char* notes = nullptr;
    std::unordered_set<uint64_t> skip;
    std::mutex skip_mutex;
};

Guard* guard() {
    static Guard* g = [] {
        auto* g = new Guard;
        VkContext* c = vk();
        const char* home = getenv("NIER_INTERNO");
        if (!c || !home) return g;
        // The list in the data folder, where it's easy to find and outlives a reinstall. The guard's own file stays in
        // the app's files: it's mapped and has to outlive the process dying, which a FUSE-backed folder can't promise.
        const char* data = getenv("NIER_DATA");
        const std::string list = std::string(data ? data : home) + "/pipeline_skip.txt";
        if (data) {  // (it was in the app's files before)
            const std::string old = std::string(home) + "/pipeline_skip.txt";
            if (access(old.c_str(), F_OK) == 0 && access(list.c_str(), F_OK) != 0 && rename(old.c_str(), list.c_str()) != 0) {
                // (another filesystem: copied)
                FILE* from = fopen(old.c_str(), "r");
                FILE* to = from ? fopen(list.c_str(), "w") : nullptr;
                if (to) {
                    char line[64];
                    while (fgets(line, sizeof line, from)) fputs(line, to);
                    fclose(to);
                    unlink(old.c_str());
                }
                if (from) fclose(from);
            }
        }
        bool older = false;
        {
            char first[64] = "";
            if (FILE* f = fopen(list.c_str(), "r")) {
                if (!fgets(first, sizeof first, f)) first[0] = 0;
                fclose(f);
            }
            first[strcspn(first, "\r\n")] = 0;
            older = first[0] && strcmp(first, kSkipHeader) != 0;
            if (strcmp(first, kSkipHeader) != 0)
                if (FILE* f = fopen(list.c_str(), "w")) {
                    if (first[0]) NVN_LOG("pipelines: the skip list was from an older translator: dropped, they're tried again\n");
                    fprintf(f, "%s\n", kSkipHeader);
                    fclose(f);
                }
        }
        // What the last run died building, before anything is marked again
        int fd = open((std::string(home) + "/pipeline_guard.bin").c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        constexpr size_t size = kGuardSlots * (sizeof(uint64_t) + kGuardNote);
        if (fd >= 0 && ftruncate(fd, size) == 0) {
            void* m = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (m != MAP_FAILED) {
                g->slots = static_cast<std::atomic<uint64_t>*>(m);
                g->notes = static_cast<char*>(m) + kGuardSlots * sizeof(uint64_t);
                if (FILE* f = fopen(list.c_str(), "a")) {
                    for (int i = 0; i < kGuardSlots; i++)
                        if (uint64_t id = g->slots[i].exchange(0); id && !older) {
                            auto* note = reinterpret_cast<GuardNote*>(g->notes + i * kGuardNote);
                            note->text[sizeof note->text - 1] = 0;
                            fprintf(f, "%016llx\n", static_cast<unsigned long long>(id));
                            if (note->vertex) fprintf(f, "%016llx\n", static_cast<unsigned long long>(note->vertex));
                            NVN_LOG("pipelines: the driver died building pipeline %016llx last time (%s): skipped from now on, "
                                    "with every pipeline of its vertex shader and formats (%016llx)\n",
                                    static_cast<unsigned long long>(id), note->text[0] ? note->text : "no vertex input",
                                    static_cast<unsigned long long>(note->vertex));
                        }
                    fclose(f);
                }
            }
        }
        if (fd >= 0) close(fd);
        if (!NVN_FLAG("no_pipeline_skip"))
            if (FILE* f = fopen(list.c_str(), "r")) {
                char line[64];
                while (fgets(line, sizeof line, f))
                    if (unsigned long long id; sscanf(line, "%llx", &id) == 1 && strncmp(line, "pipeline_skip", 13) != 0)
                        g->skip.insert(id);
                fclose(f);
            }
        if (!g->skip.empty())
            NVN_LOG("pipelines: %zu that crashed this driver are skipped (%s; delete it to try them again)\n",
                    g->skip.size(), list.c_str());
        return g;
    }();
    return g;
}

}  // namespace

static uint8_t fetch_type(uint8_t format) { return vertex_input(format).fetch; }

uint64_t vertex_input_types(const PipelineKey& key, uint64_t inputs) {
    uint64_t types = 0;
    for (unsigned i = 0; i < 16; ++i)
        if (inputs & (1ull << i)) types |= uint64_t(fetch_type(key.attribs[i].format)) << (4 * i);
    return types;
}

void normalize_vertex_input(PipelineKey& key) {
    // Keep the fetch type in the shader ABI even when Vulkan takes the buffer layout per draw.
    // (a format read wider than it is keeps its own: w is fixed in the shader)
    constexpr uint8_t representative[]{0, 0x04, 0x03, 0x56, 0x55};
    for (auto& a : key.attribs) {
        const uint8_t fetch = a.format ? fetch_type(a.format) : 0;
        a = {fetch & kFetchWIsOne ? a.format : representative[fetch], 0, 0};
    }
    memset(key.strides, 0, sizeof key.strides);
    memset(key.divisors, 0, sizeof key.divisors);
}

VkShaderStageFlagBits vk_stage(Stage s) {
    switch (s) {
        case Stage::Vertex: return VK_SHADER_STAGE_VERTEX_BIT;
        case Stage::TessControl: return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
        case Stage::TessEval: return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        case Stage::Geometry: return VK_SHADER_STAGE_GEOMETRY_BIT;
        case Stage::Fragment: return VK_SHADER_STAGE_FRAGMENT_BIT;
        default: return VK_SHADER_STAGE_COMPUTE_BIT;
    }
}

VkResult make_graphics_pipeline(const PipelineKey& key, const VkShaderModule* modules, uint64_t vs_inputs,
                                VkPipelineLayout layout, VkRenderPass rp, VkPipelineCache cache, VkPipeline& p,
                                PFN_vkCreateGraphicsPipelines create) {
    p = VK_NULL_HANDLE;
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    for (int s = 0; s < kStages; s++) {
        if (!modules[s] || s == static_cast<int>(Stage::Compute)) continue;
        VkPipelineShaderStageCreateInfo si{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        si.stage = vk_stage(static_cast<Stage>(s));
        si.module = modules[s];
        si.pName = "main";
        stages.push_back(si);
    }

    // Atributos: cada genérico usado por el vertex shader en su location; un binding por atributo
    // (cada uno con su propio buffer en el staging, ver draw()).
    std::vector<VkVertexInputAttributeDescription> attrs;
    std::vector<VkVertexInputBindingDescription> binds;
    const uint32_t static_locations = vk()->vertex_input_dynamic_state ? 0 : 16;
    for (uint32_t k = 0; k < static_locations; k++) {
        if (!(vs_inputs & (1ull << k))) continue;
        attrs.push_back({k, k, vertex_format(key.attribs[k].format), 0});
        binds.push_back({k, key.strides[k], key.divisors[k] ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX});
    }
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = static_cast<uint32_t>(binds.size());
    vi.pVertexBindingDescriptions = binds.data();
    vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
    vi.pVertexAttributeDescriptions = attrs.data();

    const int primitive = static_cast<int>(key.topology);
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = primitive == 7 ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST : vk_topology(primitive);
    ia.primitiveRestartEnable = key.primitive_restart;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = vk_polygon_mode(key.polygon.polygon_mode);
    rs.cullMode = vk_cull_mode(key.polygon.cull_face);
    // Con el viewport invertido (ver draw) el giro coincide con el de NVN: CW es CW.
    rs.frontFace = key.polygon.front_face ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
    // Depth bias where the game turns it on for this polygon mode (NVN: POINT 1, LINE 2, FILL 4); its values are
    // dynamic (cmd::PolygonOffset). <datos>/no_depth_bias at start-up: never, as before.
    static const bool no_bias = NVN_FLAG("no_depth_bias");
    rs.depthBiasEnable = !no_bias && (key.polygon.offset_enables & (1u << std::min<uint8_t>(key.polygon.polygon_mode, 2)))
                             ? VK_TRUE : VK_FALSE;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = static_cast<VkSampleCountFlagBits>(key.samples);
    ms.alphaToCoverageEnable = key.alpha_to_coverage;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    const DepthStencilState& z = key.depth_stencil;
    ds.depthTestEnable = z.depth_test;
    ds.depthWriteEnable = z.depth_write;
    ds.depthCompareOp = vk_compare_op(z.depth_func);
    ds.stencilTestEnable = z.stencil_test;
    ds.front = {vk_stencil_op(z.front_fail), vk_stencil_op(z.front_zpass), vk_stencil_op(z.front_zfail),
                vk_compare_op(z.front_func), 0xFF, 0xFF, 0};
    ds.back = {vk_stencil_op(z.back_fail), vk_stencil_op(z.back_zpass), vk_stencil_op(z.back_zfail),
               vk_compare_op(z.back_func), 0xFF, 0xFF, 0};
    std::vector<VkPipelineColorBlendAttachmentState> blends(key.color_count);
    for (uint32_t k = 0; k < key.color_count; k++) {
        const BlendState& b = key.blend[k];
        VkPipelineColorBlendAttachmentState& a = blends[k];
        a.blendEnable = (key.color.blend_enable >> k) & 1;
        a.srcColorBlendFactor = vk_blend_factor(b.src_rgb);
        a.dstColorBlendFactor = vk_blend_factor(b.dst_rgb);
        a.colorBlendOp = vk_blend_op(b.eq_rgb);
        a.srcAlphaBlendFactor = vk_blend_factor(b.src_alpha);
        a.dstAlphaBlendFactor = vk_blend_factor(b.dst_alpha);
        a.alphaBlendOp = vk_blend_op(b.eq_alpha);
        a.colorWriteMask = (key.masks.masks >> (4 * k)) & 0xF;
    }
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = key.color_count;
    cb.pAttachments = blends.data();
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
                            VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
                            VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_DEPTH_BIAS,
                            // extended dynamic state (only counted with it on)
                            VK_DYNAMIC_STATE_CULL_MODE_EXT, VK_DYNAMIC_STATE_FRONT_FACE_EXT,
                            VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE_EXT, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE_EXT,
                            VK_DYNAMIC_STATE_DEPTH_COMPARE_OP_EXT, VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE_EXT,
                            VK_DYNAMIC_STATE_STENCIL_OP_EXT};
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    // (all seven base ones: with 6, depth bias wasn't dynamic and vkCmdSetDepthBias did nothing, so the pipeline's own 0
    // was used: every offset the game asked for, on shadows and decals, was lost)
    std::vector<VkDynamicState> dynamic(dyn, dyn + (vk()->extended_dynamic_state ? std::size(dyn) : 7));
    if (vk()->vertex_input_dynamic_state) dynamic.push_back(VK_DYNAMIC_STATE_VERTEX_INPUT_EXT);
    dy.dynamicStateCount = static_cast<uint32_t>(dynamic.size());
    dy.pDynamicStates = dynamic.data();

    VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pi.stageCount = static_cast<uint32_t>(stages.size());
    pi.pStages = stages.data();
    pi.pVertexInputState = &vi;
    pi.pInputAssemblyState = &ia;
    pi.pViewportState = &vp;
    pi.pRasterizationState = &rs;
    pi.pMultisampleState = &ms;
    pi.pDepthStencilState = &ds;
    pi.pColorBlendState = &cb;
    // Con etapas de teselación Vulkan pide el estado de teselación: sin él, el driver de ARM leía patchControlPoints de
    // un puntero nulo (el crash de las Islas Heladas en un Mali-G615, dirección 0x14).
    VkPipelineTessellationStateCreateInfo ts{VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO};
    ts.patchControlPoints = std::clamp<uint32_t>(key.patch_size ? key.patch_size : 3, 1,
                                                 std::max(1u, vk()->props.limits.maxTessellationPatchSize));
    if (key.stages[static_cast<int>(Stage::TessControl)] || key.stages[static_cast<int>(Stage::TessEval)])
        pi.pTessellationState = &ts;
    pi.pDynamicState = &dy;
    pi.layout = layout;
    pi.renderPass = rp;
    // VK_INCOMPLETE con la pipeline hecha: fuera de la especificación, pero así contestan los drivers de las Adreno 5xx y
    // algunas Mali (lo mismo hace Eden): vale.
    Guard* g = guard();
    uint64_t id = 0xcbf29ce484222325ull;  // (FNV-1a of the key: zero-filled, padding included, as it's saved)
    for (size_t i = 0; i < sizeof key; i++) id = (id ^ reinterpret_cast<const uint8_t*>(&key)[i]) * 0x100000001b3ull;
    id |= 1;
    // The vertex half: the vertex shader and the format at each location it reads (not offsets or strides)
    uint64_t vertex = 0;
    if (const uint64_t vs = key.stages[static_cast<int>(Stage::Vertex)]) {
        vertex = (0x6a09e667f3bcc909ull ^ vs) * 0x100000001b3ull;
        for (uint32_t k = 0; k < 16; k++)
            if (vs_inputs >> k & 1) vertex = (vertex ^ (k << 8 | key.attribs[k].format)) * 0x100000001b3ull;
        vertex |= 1;
    }
    if (!g->skip.empty()) {
        std::lock_guard lock(g->skip_mutex);
        // (their draws are skipped like any failed pipeline's)
        if (g->skip.count(id) || (vertex && g->skip.count(vertex))) return VK_ERROR_INITIALIZATION_FAILED;
    }
    int slot = -1;
    if (g->slots)
        for (int i = 0; i < kGuardSlots && slot < 0; i++) {
            uint64_t free = 0;
            if (g->slots[i].compare_exchange_strong(free, id)) slot = i;
        }
    if (slot >= 0) {
        // Its vertex input, for the log of the next run if the driver dies in here (formats, offset, stream stride)
        auto* note = reinterpret_cast<GuardNote*>(g->notes + slot * kGuardNote);
        constexpr int room = sizeof note->text;
        int n = 0;
        note->vertex = vertex;
        note->text[0] = 0;
        for (uint32_t k = 0; k < 16 && n < room - 40; k++) {
            if (!(vs_inputs >> k & 1)) continue;
            const FormatInfo* f = format_info(key.attribs[k].format);
            n += snprintf(note->text + n, room - n, "%s%u %s+%u/%u", n ? ", " : "", k, f ? f->name : "?",
                          unsigned(key.attribs[k].offset), unsigned(key.strides[k]));
        }
    }
    const VkResult r = create(vk()->device, cache, 1, &pi, nullptr, &p);
    if (slot >= 0) g->slots[slot].store(0, std::memory_order_release);
    return r == VK_INCOMPLETE && p ? VK_SUCCESS : r;
}

VkResult make_compute_pipeline(VkShaderModule module, VkPipelineLayout layout, VkPipelineCache cache, VkPipeline& p) {
    p = VK_NULL_HANDLE;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = module;
    ci.stage.pName = "main";
    ci.layout = layout;
    return vkCreateComputePipelines(vk()->device, cache, 1, &ci, nullptr, &p);
}

}  // namespace nvn
