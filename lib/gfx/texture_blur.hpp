#pragma once

#include "texture.hpp"

namespace aurora::gfx::texture_blur {
struct Request {
  TextureHandle temporary;
  wgpu::Buffer params;
};

void initialize();
void shutdown();
void clear_cache();
Request prepare(const TextureHandle& texture, uint32_t radius);
void run(const wgpu::CommandEncoder& encoder, const TextureHandle& texture, const Request& request);
} // namespace aurora::gfx::texture_blur
