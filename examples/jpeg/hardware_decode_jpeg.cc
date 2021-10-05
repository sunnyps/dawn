#include <cstdint>
#include <vector>

#include <dawn/webgpu_cpp.h>

#include "examples/jpeg/hardware_decode_jpeg.h"
#include "utils/WGPUHelpers.h"

wgpu::TextureView HardwareDecodeJpeg(wgpu::Device device, const JpegData& jpeg) {
    // TODO: Move YUV conversion, IDCT computation, and maybe DCT decode into compute stages.

    IntBlockMap dct_blocks = DecodeDCTBlocks(jpeg);
    IntBlockMap color_blocks = PerformIDCT(dct_blocks);

    uint32_t stride;
    std::vector<uint32_t> rgba = ConvertYCbCrToRGBA(jpeg, color_blocks, &stride);

    wgpu::Buffer uniforms = utils::CreateBufferFromData(device, wgpu::BufferUsage::Uniform,
                                                        {stride / 4, jpeg.width, jpeg.height});

    wgpu::Buffer image_data = utils::CreateBufferFromData(
        device, rgba.data(), rgba.size() * sizeof(uint32_t), wgpu::BufferUsage::Storage);

    wgpu::TextureDescriptor descriptor;
    descriptor.dimension = wgpu::TextureDimension::e2D;
    descriptor.size.width = jpeg.width;
    descriptor.size.height = jpeg.height;
    descriptor.size.depthOrArrayLayers = 1;
    descriptor.sampleCount = 1;
    descriptor.format = wgpu::TextureFormat::RGBA8Unorm;
    descriptor.mipLevelCount = 1;
    descriptor.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::StorageBinding;
    wgpu::Texture texture = device.CreateTexture(&descriptor);

    wgpu::ShaderModule compute_shader = utils::CreateShaderModule(device, R"(
        [[block]] struct Params {
          stride: u32;
          width: u32;
          height: u32;
        };

        [[block]] struct ImageData {
          colors: array<u32>;
        };

        [[group(0), binding(0)]] var<uniform> params: Params;
        [[group(0), binding(1)]] var<storage, read> image_data: ImageData;

        [[group(0), binding(2)]]
        var decoded_image: texture_storage_2d<rgba8unorm, write>;

        [[stage(compute), workgroup_size(1)]] fn main(
            [[builtin(global_invocation_id)]] id: vec3<u32>) {
          if (id.x >= params.width || id.y >= params.height) {
            return;
          }

          // "Decode" i.e. copy the raw data from the input buffer to the output texture.
          let rgba = image_data.colors[id.y * params.stride + id.x];
          let r = rgba & 0xffu;
          let g = (rgba >> 8u) & 0xffu;
          let b = (rgba >> 16u) & 0xffu;
          let a = rgba >> 24u;
          textureStore(decoded_image, vec2<i32>(i32(id.x), i32(id.y)),
                       vec4<f32>(vec4<u32>(r, g, b, a)) / 255.0);
        }
    )");

    wgpu::ComputePipelineDescriptor pipeline_descriptor;
    pipeline_descriptor.compute.module = compute_shader;
    pipeline_descriptor.compute.entryPoint = "main";
    wgpu::ComputePipeline pipeline = device.CreateComputePipeline(&pipeline_descriptor);

    wgpu::BindGroup bg = utils::MakeBindGroup(device, pipeline.GetBindGroupLayout(0),
                                              {
                                                  {0, uniforms},
                                                  {1, image_data},
                                                  {2, texture.CreateView()},
                                              });

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bg);
    pass.Dispatch(jpeg.width, jpeg.height);
    pass.EndPass();
    wgpu::CommandBuffer commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);

    return texture.CreateView();
}
