#include <cstdint>
#include <cstring>
#include <vector>

#include <dawn/webgpu_cpp.h>

#include "examples/jpeg/hardware_decode_jpeg.h"
#include "utils/WGPUHelpers.h"

std::vector<int32_t> FlattenDCTBlocks(const IntBlockList& blocks) {
    std::vector<int32_t> data;
    data.resize(blocks.size() * 64);
    int32_t* out_data = data.data();
    for (const IntBlock& block : blocks) {
        memcpy(out_data, block.data(), block.size() * sizeof(int32_t));
        out_data += block.size();
    }
    return data;
}

wgpu::TextureView HardwareDecodeJpeg(wgpu::Device device, const JpegData& jpeg) {
    // TODO: Move YUV conversion, IDCT computation, and maybe DCT decode into compute stages.

    IntBlockMap dct_blocks = DecodeDCTBlocks(jpeg);

    std::vector<int32_t> y = FlattenDCTBlocks(dct_blocks[1]);
    std::vector<int32_t> cr = FlattenDCTBlocks(dct_blocks[2]);
    std::vector<int32_t> cb = FlattenDCTBlocks(dct_blocks[3]);

    wgpu::Buffer uniforms =
        utils::CreateBufferFromData(device, wgpu::BufferUsage::Uniform, {jpeg.width, jpeg.height});
    wgpu::Buffer y_dct_data = utils::CreateBufferFromData(
        device, y.data(), y.size() * sizeof(int32_t), wgpu::BufferUsage::Storage);
    wgpu::Buffer cr_dct_data = utils::CreateBufferFromData(
        device, cr.data(), cr.size() * sizeof(int32_t), wgpu::BufferUsage::Storage);
    wgpu::Buffer cb_dct_data = utils::CreateBufferFromData(
        device, cb.data(), cb.size() * sizeof(int32_t), wgpu::BufferUsage::Storage);

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
          width: u32;
          height: u32;
        };

        [[block]] struct DCTData {
          data: array<i32>;
        };

        [[group(0), binding(0)]] var<uniform> params: Params;

        [[group(0), binding(1)]]
        var decoded_image: texture_storage_2d<rgba8unorm, write>;

        [[group(1), binding(0)]] var<storage, read> y_dct: DCTData;
        [[group(1), binding(1)]] var<storage, read> cb_dct: DCTData;
        [[group(1), binding(2)]] var<storage, read> cr_dct: DCTData;

        var<private> kZigZagTable: array<u32, 64> = array<u32, 64>(
            0u,  1u,  5u,  6u,  14u, 15u, 27u, 28u, 2u,  4u,  7u,  13u, 16u, 26u, 29u, 42u, 3u,  8u,
            12u, 17u, 25u, 30u, 41u, 43u, 9u,  11u, 18u, 24u, 31u, 40u, 44u, 53u, 10u, 19u, 23u,
            32u, 39u, 45u, 52u, 54u, 20u, 22u, 33u, 38u, 46u, 51u, 55u, 60u, 21u, 34u, 37u, 47u,
            50u, 56u, 59u, 61u, 35u, 36u, 48u, 49u, 57u, 58u, 62u, 63u);

        var<private> kIDCTCoefficients: array<f32, 64> = array<f32, 64>(
            0.7071067811865475, 0.7071067811865475, 0.7071067811865475, 0.7071067811865475,
            0.7071067811865475, 0.7071067811865475, 0.7071067811865475, 0.7071067811865475,
            0.9807852804032304, 0.8314696123025452, 0.5555702330196023, 0.19509032201612833,
            -0.1950903220161282, -0.555570233019602, -0.8314696123025453, -0.9807852804032304,
            0.9238795325112867, 0.38268343236508984, -0.3826834323650897, -0.9238795325112867,
            -0.9238795325112868, -0.38268343236509034, 0.38268343236509, 0.9238795325112865,
            0.8314696123025452, -0.1950903220161282, -0.9807852804032304, -0.5555702330196022,
            0.5555702330196018, 0.9807852804032304, 0.19509032201612878, -0.8314696123025451,
            0.7071067811865476, -0.7071067811865475, -0.7071067811865477, 0.7071067811865474,
            0.7071067811865477, -0.7071067811865467, -0.7071067811865471, 0.7071067811865466,
            0.5555702330196023, -0.9807852804032304, 0.1950903220161283, 0.8314696123025455,
            -0.8314696123025451, -0.19509032201612803, 0.9807852804032307, -0.5555702330196015,
            0.38268343236508984, -0.9238795325112868, 0.9238795325112865, -0.3826834323650899,
            -0.38268343236509056, 0.9238795325112867, -0.9238795325112864, 0.38268343236508956,
            0.19509032201612833, -0.5555702330196022, 0.8314696123025455, -0.9807852804032307,
            0.9807852804032304, -0.831469612302545, 0.5555702330196015, -0.19509032201612858);

        [[stage(compute), workgroup_size(8, 8)]] fn main(
            [[builtin(global_invocation_id)]] global_id: vec3<u32>,
            [[builtin(workgroup_id)]] block_id: vec3<u32>,
            [[builtin(local_invocation_id)]] local_id: vec3<u32>) {
          if (global_id.x >= params.width || global_id.y >= params.height) {
            return;
          }

          let block_base_data_index = block_id.x * 8u + block_id.y * params.width * 8u;
          var y_sum: f32 = 0.0;
          var cb_sum: f32 = 0.0;
          var cr_sum: f32 = 0.0;
          for (var y: u32 = 0u; y < 8u; y = y + 1u) {
            let row_coefficient = kIDCTCoefficients[local_id.y + y * 8u];
            for (var x: u32 = 0u; x < 8u; x = x + 1u) {
              let block_index = x + y * 8u;
              let zigzagged_block_index = kZigZagTable[block_index];
              let zigzagged_x = zigzagged_block_index & 7u;
              let zigzagged_y = zigzagged_block_index / 8u;
              let data_index = block_base_data_index + zigzagged_x + zigzagged_y * params.width;
              let c = row_coefficient * kIDCTCoefficients[local_id.x + x * 8u];
              y_sum = fma(f32(y_dct.data[data_index]), c, y_sum);
              cb_sum = fma(f32(cb_dct.data[data_index]), c, cb_sum);
              cr_sum = fma(f32(cr_dct.data[data_index]), c, cr_sum);
            }
          }

          let y = y_sum / 4.0;
          let cb = cb_sum / 4.0;
          let cr = cr_sum / 4.0;
          let kr = 1.402 * cr + y;
          let kb = 1.772 * cb + y;
          let kg = (y - 0.114 * kb - 0.299 * kr) / 0.587;
          let r = clamp(kr + 128.0, 0.0, 255.0);
          let g = clamp(kg + 128.0, 0.0, 255.0);
          let b = clamp(kb + 128.0, 0.0, 255.0);
          textureStore(decoded_image, vec2<i32>(i32(global_id.x), i32(global_id.y)),
                       vec4<f32>(r, g, b, 255.0) / 255.0);
        }
    )");

    wgpu::ComputePipelineDescriptor pipeline_descriptor;
    pipeline_descriptor.compute.module = compute_shader;
    pipeline_descriptor.compute.entryPoint = "main";
    wgpu::ComputePipeline pipeline = device.CreateComputePipeline(&pipeline_descriptor);

    wgpu::BindGroup bg0 = utils::MakeBindGroup(device, pipeline.GetBindGroupLayout(0),
                                               {
                                                   {0, uniforms},
                                                   {1, texture.CreateView()},
                                               });
    wgpu::BindGroup bg1 = utils::MakeBindGroup(device, pipeline.GetBindGroupLayout(1),
                                               {
                                                   {0, y_dct_data},
                                                   {1, cb_dct_data},
                                                   {2, cr_dct_data},
                                               });

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bg0);
    pass.SetBindGroup(1, bg1);
    pass.Dispatch(jpeg.width / 8, jpeg.height / 8);
    pass.EndPass();
    wgpu::CommandBuffer commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);

    return texture.CreateView();
}
