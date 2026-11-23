#pragma once

#include "nnsdk/nvn/objects.h"
#include "nnsdk/nvn/state.h"
#include <cstring>

namespace nvn {

// Solo contenido (identidades de lo traducido, formatos del pase, estado), sin direcciones ni handles de esta partida: los
// programas con el mismo SPIR-V comparten pipelines, y la clave se puede guardar y crear en la partida siguiente
// (precalentado, ver PipelineRecord). Se rellena tras un memset a cero (el relleno también se hashea y se guarda).
struct PipelineKey {
    uint64_t stages[kStages];  // ProgramStage::spirv_id de cada etapa (0: sin ella); en las de compute solo la suya
    // El pase con el que se crea (el de LOAD/STORE de render_pass: lo que cuenta para la compatibilidad).
    uint32_t pass_colors[8];   // VkFormat de cada render target (VK_FORMAT_UNDEFINED: sin él)
    uint32_t pass_depth;       // VkFormat de la profundidad
    uint32_t topology;         // primitiva de NVN; kComputeTopology en las de compute
    PolygonState polygon;
    DepthStencilState depth_stencil;
    ColorState color;
    ChannelMaskState masks;
    BlendState blend[8];
    uint32_t color_count, samples;
    uint8_t alpha_to_coverage;  // estaba en el pipeline pero no en la clave
    uint8_t skip_gs;            // pre-pase sin su geometry shader de compactación
    uint8_t primitive_restart;  // solo en las topologías de tiras y abanicos (las de listas no lo admiten)
    uint8_t patch_size;         // con teselación: puntos de control por parche (iba en el relleno: 0 en las guardadas)
    VertexAttribState attribs[16];
    uint32_t strides[16], divisors[16];
};
constexpr uint32_t kComputeTopology = ~0u;
uint64_t vertex_input_types(const PipelineKey& key, uint64_t inputs);
void normalize_vertex_input(PipelineKey& key);

// With extended dynamic state: the key without what's set per draw
inline void normalize_dynamic(PipelineKey& key) {
    key.polygon.cull_face = key.polygon.front_face = 0;
    memset(&key.depth_stencil, 0, sizeof key.depth_stencil);
}
VkShaderStageFlagBits vk_stage(nvnshader::Stage s);
VkResult make_graphics_pipeline(const PipelineKey& key, const VkShaderModule* modules, uint64_t vs_inputs,
                                VkPipelineLayout layout, VkRenderPass pass, VkPipelineCache cache, VkPipeline& pipeline,
                                PFN_vkCreateGraphicsPipelines create = vkCreateGraphicsPipelines);
VkResult make_compute_pipeline(VkShaderModule module, VkPipelineLayout layout, VkPipelineCache cache, VkPipeline& pipeline);

}  // namespace nvn
