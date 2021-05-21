// Copyright 2018 The Dawn Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef UTILS_COMBORENDERPIPELINEDESCRIPTOR_H_
#define UTILS_COMBORENDERPIPELINEDESCRIPTOR_H_

#include <dawn/webgpu_cpp.h>

#include "common/Constants.h"

#include <array>

namespace utils {

    // Primarily used by tests to easily set up the vertex buffer state portion of a RenderPipeline.
    class ComboVertexStateDescriptor {
      public:
        ComboVertexStateDescriptor();

        ComboVertexStateDescriptor(const ComboVertexStateDescriptor&) = delete;
        ComboVertexStateDescriptor& operator=(const ComboVertexStateDescriptor&) = delete;
        ComboVertexStateDescriptor(ComboVertexStateDescriptor&&) = delete;
        ComboVertexStateDescriptor& operator=(ComboVertexStateDescriptor&&) = delete;

        uint32_t vertexBufferCount;
        std::array<wgpu::VertexBufferLayoutDescriptor, kMaxVertexBuffers> cVertexBuffers;
        std::array<wgpu::VertexAttributeDescriptor, kMaxVertexAttributes> cAttributes;
    };

    class ComboRenderPipelineDescriptor : public wgpu::RenderPipelineDescriptor {
      public:
        ComboRenderPipelineDescriptor();

        ComboRenderPipelineDescriptor(const ComboRenderPipelineDescriptor&) = delete;
        ComboRenderPipelineDescriptor& operator=(const ComboRenderPipelineDescriptor&) = delete;
        ComboRenderPipelineDescriptor(ComboRenderPipelineDescriptor&&) = delete;
        ComboRenderPipelineDescriptor& operator=(ComboRenderPipelineDescriptor&&) = delete;

        wgpu::DepthStencilState* EnableDepthStencil(
            wgpu::TextureFormat format = wgpu::TextureFormat::Depth24PlusStencil8);

        std::array<wgpu::VertexBufferLayout, kMaxVertexBuffers> cBuffers;
        std::array<wgpu::VertexAttribute, kMaxVertexAttributes> cAttributes;
        std::array<wgpu::ColorTargetState, kMaxColorAttachments> cTargets;
        std::array<wgpu::BlendState, kMaxColorAttachments> cBlends;

        wgpu::FragmentState cFragment;
        wgpu::DepthStencilState cDepthStencil;
    };

}  // namespace utils

#endif  // UTILS_COMBORENDERPIPELINEDESCRIPTOR_H_
