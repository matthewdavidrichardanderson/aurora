#include "texture_blur.hpp"

#include "../webgpu/gpu.hpp"
#include "../webgpu/gpu_prof.hpp"

#include <array>
#include <cstring>
#include <unordered_map>

namespace aurora::gfx::texture_blur {
namespace {
using webgpu::g_device;

wgpu::BindGroupLayout g_bindGroupLayout;
wgpu::RenderPipeline g_horizontalPipeline;
wgpu::RenderPipeline g_verticalPipeline;
std::unordered_map<uint64_t, TextureHandle> g_temporaryTextures;
std::unordered_map<uint32_t, wgpu::Buffer> g_parameterBuffers;

struct Params {
  uint32_t radius;
  uint32_t padding[3];
};

constexpr std::string_view Shader = R"(
struct Params {
  radius: u32,
  padding0: u32,
  padding1: u32,
  padding2: u32,
};

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@vertex fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4f {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, 1.0),
    vec2f(-1.0, -3.0),
    vec2f(3.0, 1.0)
  );
  return vec4f(positions[vertex_index], 0.0, 1.0);
}

fn blur(position: vec4f, horizontal: bool) -> vec4f {
  let initial_coords = vec2i(position.xy);
  let dimensions = vec2i(textureDimensions(source_texture));
  let coordinate = select(initial_coords.y, initial_coords.x, horizontal);
  let length = select(dimensions.y, dimensions.x, horizontal);
  let radius = i32(params.radius);
  var color = vec4f(0.0);
  var count = 0.0;

  for (var offset = -radius; offset <= radius; offset += 1) {
    let sample_position = coordinate + offset;
    if (sample_position >= 0 && sample_position <= length) {
      let sample_coords =
        select(vec2i(initial_coords.x, sample_position),
               vec2i(sample_position, initial_coords.y), horizontal);
      color += textureLoad(source_texture, sample_coords, 0);
      count += 1.0;
    }
  }
  return color / count;
}

@fragment fn fs_horizontal(@builtin(position) position: vec4f) -> @location(0) vec4f {
  // This filter is applied to three TP bloom copies. Applying the cube root here
  // produces an overall brightness multiplier of 0.6 instead of 0.6 cubed.
  return blur(position, true) * 0.84343267;
}

@fragment fn fs_vertical(@builtin(position) position: vec4f) -> @location(0) vec4f {
  return blur(position, false);
}
)";

wgpu::RenderPipeline create_pipeline(const wgpu::ShaderModule& module, const wgpu::PipelineLayout& layout,
                                     const char* label, const char* fragmentEntryPoint) {
  const std::array colorTargets{
      wgpu::ColorTargetState{
          .format = webgpu::g_graphicsConfig.surfaceConfiguration.format,
          .writeMask = wgpu::ColorWriteMask::All,
      },
  };
  const wgpu::FragmentState fragmentState{
      .module = module,
      .entryPoint = fragmentEntryPoint,
      .targetCount = colorTargets.size(),
      .targets = colorTargets.data(),
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = label,
      .layout = layout,
      .vertex =
          wgpu::VertexState{
              .module = module,
              .entryPoint = "vs_main",
          },
      .primitive =
          wgpu::PrimitiveState{
              .topology = wgpu::PrimitiveTopology::TriangleList,
          },
      .fragment = &fragmentState,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

wgpu::BindGroup create_bind_group(const TextureHandle& texture, const wgpu::Buffer& params) {
  const std::array entries{
      wgpu::BindGroupEntry{
          .binding = 0,
          .textureView = texture->sampleTextureView,
      },
      wgpu::BindGroupEntry{
          .binding = 1,
          .buffer = params,
          .size = sizeof(Params),
      },
  };
  const wgpu::BindGroupDescriptor descriptor{
      .label = "Bloom blur bind group",
      .layout = g_bindGroupLayout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  return g_device.CreateBindGroup(&descriptor);
}

void draw_pass(const wgpu::CommandEncoder& encoder, const char* label, const TextureHandle& destination,
               const wgpu::RenderPipeline& pipeline, const wgpu::BindGroup& bindGroup) {
  const std::array colorAttachments{
      wgpu::RenderPassColorAttachment{
          .view = destination->attachmentTextureView,
          .loadOp = wgpu::LoadOp::Clear,
          .storeOp = wgpu::StoreOp::Store,
          .clearValue = {0.0, 0.0, 0.0, 0.0},
      },
  };
  const wgpu::RenderPassDescriptor descriptor{
      .label = label,
      .colorAttachmentCount = colorAttachments.size(),
      .colorAttachments = colorAttachments.data(),
      .timestampWrites = webgpu::gpu_prof::pass_writes(label),
  };
  const auto pass = encoder.BeginRenderPass(&descriptor);
  pass.SetPipeline(pipeline);
  pass.SetBindGroup(0, bindGroup);
  pass.Draw(3);
  pass.End();
}
} // namespace

void initialize() {
  wgpu::ShaderSourceWGSL shaderSource{};
  shaderSource.code = Shader;
  const wgpu::ShaderModuleDescriptor moduleDescriptor{
      .nextInChain = &shaderSource,
      .label = "Bloom blur shader",
  };
  const auto module = g_device.CreateShaderModule(&moduleDescriptor);

  const std::array layoutEntries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture =
              wgpu::TextureBindingLayout{
                  .sampleType = wgpu::TextureSampleType::Float,
                  .viewDimension = wgpu::TextureViewDimension::e2D,
              },
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Fragment,
          .buffer =
              wgpu::BufferBindingLayout{
                  .type = wgpu::BufferBindingType::Uniform,
                  .minBindingSize = sizeof(Params),
              },
      },
  };
  const wgpu::BindGroupLayoutDescriptor bindGroupLayoutDescriptor{
      .label = "Bloom blur bind group layout",
      .entryCount = layoutEntries.size(),
      .entries = layoutEntries.data(),
  };
  g_bindGroupLayout = g_device.CreateBindGroupLayout(&bindGroupLayoutDescriptor);

  const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{
      .label = "Bloom blur pipeline layout",
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &g_bindGroupLayout,
  };
  const auto pipelineLayout = g_device.CreatePipelineLayout(&pipelineLayoutDescriptor);
  g_horizontalPipeline =
      create_pipeline(module, pipelineLayout, "Bloom blur horizontal pipeline", "fs_horizontal");
  g_verticalPipeline =
      create_pipeline(module, pipelineLayout, "Bloom blur vertical pipeline", "fs_vertical");
}

void shutdown() {
  clear_cache();
  g_verticalPipeline = {};
  g_horizontalPipeline = {};
  g_bindGroupLayout = {};
}

void clear_cache() {
  g_parameterBuffers.clear();
  g_temporaryTextures.clear();
}

Request prepare(const TextureHandle& texture, uint32_t radius) {
  auto& paramsBuffer = g_parameterBuffers[radius];
  if (!paramsBuffer) {
    const Params params{.radius = radius};
    const wgpu::BufferDescriptor paramsDescriptor{
        .label = "Bloom blur parameters",
        .usage = wgpu::BufferUsage::Uniform,
        .size = sizeof(Params),
        .mappedAtCreation = true,
    };
    paramsBuffer = g_device.CreateBuffer(&paramsDescriptor);
    std::memcpy(paramsBuffer.GetMappedRange(0, sizeof(params)), &params, sizeof(params));
    paramsBuffer.Unmap();
  }

  const uint64_t textureKey =
      (static_cast<uint64_t>(texture->size.width) << 32) | texture->size.height;
  auto& temporary = g_temporaryTextures[textureKey];
  if (!temporary) {
    temporary =
        new_render_texture(texture->size.width, texture->size.height, texture->gxFormat,
                           "Bloom blur temporary texture");
  }

  return {
      .temporary = temporary,
      .params = paramsBuffer,
  };
}

void run(const wgpu::CommandEncoder& encoder, const TextureHandle& texture, const Request& request) {
  const webgpu::gpu_prof::Zone zone{encoder, "Bloom blur"};
  const auto horizontalBindGroup = create_bind_group(texture, request.params);
  draw_pass(encoder, "Bloom blur horizontal", request.temporary, g_horizontalPipeline,
            horizontalBindGroup);

  const auto verticalBindGroup = create_bind_group(request.temporary, request.params);
  draw_pass(encoder, "Bloom blur vertical", texture, g_verticalPipeline, verticalBindGroup);
}
} // namespace aurora::gfx::texture_blur
