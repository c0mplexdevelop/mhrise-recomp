#pragma once

#include "nnsdk/nvn/commands.h"

namespace nvn {

constexpr int kMaxUniforms = 18, kMaxTextures = 32, kMaxStorage = 16, kMaxVertexBuffers = 16, kMaxImages = 16;

struct StageBindings {
    BufferRange uniforms[kMaxUniforms]{};
    BufferRange storage[kMaxStorage]{};
    uint64_t textures[kMaxTextures]{};
    bool combined[kMaxTextures]{};
    // Las combinadas (BindTexture, textura | sampler) van en NVN a su propia tabla, la baja de c2 (0x20 + 8·i), y las
    // separadas (BindSeparateTexture) a la de 0x168 + 8·i: son dos ranuras 0 distintas. MH Rise enlaza separadas para todo
    // menos los iconos de la interfaz (68 de 8085 texturas en tres trazas del 5-11): en un solo array, el icono de una
    // fila de la tabla de recompensas pisaba la máscara de la lista en la ranura 0 y las filas siguientes leían su propia
    // máscara recién borrada (cero): no se veían. Ver tablas_separadas_.
    uint64_t combined_textures[kMaxTextures]{};
    uint64_t samplers[kMaxTextures]{};
    uint64_t images[kMaxImages]{};  // handles de BindImage (= id en el pool de texturas; 0: nada)
};

struct QueueState {
    cmd::SetRenderTargets rt{};
    VkViewport viewport{0, 0, 1280, 720, 0, 1};
    VkRect2D scissor{};
    bool scissor_set = false;
    const ProgramStage* stages[kStages] = {};  // por nvnshader::Stage
    uint64_t stage_generation[kStages] = {};
    // NVN stage bits (BindProgram) whose bound program has a stage still translating in the background
    // (translate_or_queue): draws wait until it's ready.
    uint32_t pending_stages = 0;
    BlendState blend[8]{};
    ColorState color{};
    ChannelMaskState masks{0xFFFFFFFF};
    DepthStencilState depth_stencil{};
    PolygonState polygon{};
    MultisampleState multisample{};
    std::vector<VertexAttribState> attribs;
    std::vector<VertexStreamState> streams;
    BufferRange vertex_buffers[kMaxVertexBuffers]{};
    StageBindings bindings[kStages];
    ResourceRef<DescriptorPool> texture_pool;
    ResourceRef<DescriptorPool> sampler_pool;
    uint32_t stencil_ref[2] = {}, stencil_mask[2] = {0xFF, 0xFF}, stencil_value_mask[2] = {0xFF, 0xFF};
    float depth_bias[3] = {};  // units, clamp, slope factor (vkCmdSetDepthBias order)
    float blend_color[4] = {};
    bool render_enabled = true;
    bool primitive_restart = false;
    uint32_t restart_index = ~0u;
    uint32_t patch_size = 3;  // (lo de OpenGL; MH Rise pone el suyo antes de dibujar)
    ProgramStage* compute = nullptr;
};

}  // namespace nvn
