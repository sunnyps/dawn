#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "GLFW/glfw3.h"

#include "examples/SampleUtils.h"
#include "examples/jpeg/hardware_jpeg_decoder.h"
#include "examples/jpeg/jpeg_decoder.h"
#include "examples/jpeg/software_jpeg_decoder.h"
#include "utils/ComboRenderPipelineDescriptor.h"
#include "utils/ScopedAutoreleasePool.h"
#include "utils/SystemUtils.h"
#include "utils/WGPUHelpers.h"

// Creates a simple render pipeline to render a textured quad with a UBO-given size.
wgpu::RenderPipeline CreateRenderPipeline(wgpu::Device device) {
    wgpu::ShaderModule vertex_shader = utils::CreateShaderModule(device, R"(
        [[block]] struct Params {
          width: f32;
          height: f32;
        };
        struct VertexOut {
          [[builtin(position)]] position: vec4<f32>;
          [[location(0)]] uv: vec2<f32>;
        };
        [[group(0), binding(0)]] var<uniform> params: Params;
        [[stage(vertex)]]
        fn main([[builtin(vertex_index)]] index: u32) -> VertexOut {
            let w = params.width;
            let h = params.height;
            switch (index) {
              case 0u: {
                return VertexOut(vec4<f32>(w, -h, 0.0, 1.0), vec2<f32>(1.0, 1.0));
              }
              case 1u: {
                return VertexOut(vec4<f32>(w, h, 0.0, 1.0), vec2<f32>(1.0, 0.0));
              }
              case 2u: {
                return VertexOut(vec4<f32>(-w, -h, 0.0, 1.0), vec2<f32>(0.0, 1.0));
              }
              case 3u: {
                return VertexOut(vec4<f32>(-w, h, 0.0, 1.0), vec2<f32>(0.0, 0.0));
              }
              default: {
                break;
              }
            }
            return VertexOut(vec4<f32>(0.0), vec2<f32>(0.0));
        }
    )");

    wgpu::ShaderModule fragment_shader = utils::CreateShaderModule(device, R"(
        [[group(0), binding(1)]] var jpeg_texture: texture_2d<f32>;
        [[group(0), binding(2)]] var jpeg_sampler: sampler;
        [[stage(fragment)]]
        fn main([[location(0)]] uv: vec2<f32>) -> [[location(0)]] vec4<f32> {
            return textureSample(jpeg_texture, jpeg_sampler, uv);
        }
    )");

    utils::ComboRenderPipelineDescriptor descriptor;
    descriptor.vertex.module = vertex_shader;
    descriptor.primitive.topology = wgpu::PrimitiveTopology::TriangleStrip;
    descriptor.primitive.stripIndexFormat = wgpu::IndexFormat::Uint16;
    descriptor.cFragment.module = fragment_shader;
    descriptor.cTargets[0].format = GetPreferredSwapChainTextureFormat();
    return device.CreateRenderPipeline(&descriptor);
}

std::vector<uint8_t> ReadFile(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        return std::vector<uint8_t>(std::istreambuf_iterator<char>(file), {});
    }
    return {};
}

// Computes rendered quad size based on image size and window size. This stretches the quad to fit
// the largest window dimension possible while preserving the image's aspect ratio.
void UpdateRenderDimensions(wgpu::Queue queue,
                            wgpu::Buffer uniforms,
                            int width,
                            int height,
                            int window_width,
                            int window_height) {
    const float x_stretch = static_cast<float>(window_width) / width;
    const float y_stretch = static_cast<float>(window_height) / height;
    float dimensions[2];
    if (x_stretch * height <= window_height) {
        dimensions[0] = 1.0f;
        dimensions[1] = x_stretch * static_cast<float>(height * height) /
                        static_cast<float>(width * window_height);
    } else {
        dimensions[0] = y_stretch * static_cast<float>(width * width) /
                        static_cast<float>(height * window_width);
        dimensions[1] = 1.0f;
    }
    queue.WriteBuffer(uniforms, 0, reinterpret_cast<uint8_t*>(&dimensions[0]), sizeof(dimensions));
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [-sw|-hw] <filename>\n";
        return 1;
    }

    if (!InitSample(argc, argv)) {
        return 1;
    }

    bool force_software = false;
    bool force_hardware = false;
    for (int i = 1; i < argc - 1; ++i) {
        if (argv[i] == std::string("-sw")) {
            force_software = true;
        } else if (argv[i] == std::string("-hw")) {
            force_hardware = true;
        }
    }

    wgpu::Device device = CreateCppDawnDevice();
    wgpu::Queue queue = device.GetQueue();

    std::unique_ptr<JpegDecoder> decoder;
    const bool use_software = force_software || !force_hardware;
    if (use_software) {
        decoder = std::make_unique<SoftwareJpegDecoder>(device);
    } else {
        decoder = std::make_unique<HardwareJpegDecoder>(device);
    }

    wgpu::Buffer render_uniforms = utils::CreateBufferFromData(
        device, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, {0.0f, 0.0f});

    wgpu::SwapChain swapchain = GetSwapChain(device);
    wgpu::RenderPipeline render_pipeline = CreateRenderPipeline(device);

    swapchain.Configure(GetPreferredSwapChainTextureFormat(), wgpu::TextureUsage::RenderAttachment,
                        640, 480);

    std::vector<uint8_t> raw_jpeg_data = ReadFile(argv[argc - 1]);
    if (raw_jpeg_data.empty()) {
      std::cerr << "nah\n";
      return 1;
    }

    wgpu::Sampler sampler = device.CreateSampler();

    int last_window_width = 0;
    int last_window_height = 0;
    while (!ShouldQuit()) {
        int width;
        int height;
        wgpu::TextureView decoded_image = decoder->Decode(raw_jpeg_data, &width, &height);
        if (!decoded_image) {
            std::cerr << "Failed to decode!\n";
            return 1;
        }

        wgpu::BindGroup bg = utils::MakeBindGroup(device, render_pipeline.GetBindGroupLayout(0),
                                                  {
                                                      {0, render_uniforms},
                                                      {1, decoded_image},
                                                      {2, sampler},
                                                  });

        utils::ScopedAutoreleasePool pool;

        int window_width, window_height;
        glfwGetWindowSize(GetGLFWWindow(), &window_width, &window_height);

        if (last_window_width != window_width || last_window_height != window_height) {
            last_window_width = window_width;
            last_window_height = window_height;
            swapchain.Configure(GetPreferredSwapChainTextureFormat(),
                                wgpu::TextureUsage::RenderAttachment, last_window_width,
                                last_window_height);
            UpdateRenderDimensions(queue, render_uniforms, width, height, window_width, window_height);
        }

        wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
        utils::ComboRenderPassDescriptor descriptor({swapchain.GetCurrentTextureView()});
        wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&descriptor);
        pass.SetPipeline(render_pipeline);
        pass.SetBindGroup(0, bg);
        pass.Draw(4);
        pass.EndPass();
        wgpu::CommandBuffer commands = encoder.Finish();
        queue.Submit(1, &commands);

        swapchain.Present();
        DoFlush();
        utils::USleep(16000);
    }

    return 0;
}
