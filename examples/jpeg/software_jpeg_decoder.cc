#include "examples/jpeg/software_jpeg_decoder.h"

#include <cstdint>
#include <iostream>
#include <dawn/webgpu_cpp.h>

#include "examples/jpeg/jpeg_data.h"

SoftwareJpegDecoder::SoftwareJpegDecoder(wgpu::Device device)
    : device_(device) {}

SoftwareJpegDecoder::~SoftwareJpegDecoder() = default;

wgpu::TextureView SoftwareJpegDecoder::Decode(std::vector<uint8_t> data, int* width, int* height) {
    JpegData jpeg;
    if (!ParseJpegData(std::move(data), jpeg)) {
      return {};
    }

    *width = static_cast<int>(jpeg.width);
    *height = static_cast<int>(jpeg.height);

    IntBlockMap dct_blocks = DecodeDCTBlocks(jpeg);
    IntBlockMap color_blocks = PerformIDCT(dct_blocks);

    uint32_t stride;
    std::vector<uint32_t> rgba = ConvertYCbCrToRGBA(jpeg, color_blocks, &stride);

    wgpu::TextureDescriptor descriptor;
    descriptor.dimension = wgpu::TextureDimension::e2D;
    descriptor.size.width = jpeg.width;
    descriptor.size.height = jpeg.height;
    descriptor.size.depthOrArrayLayers = 1;
    descriptor.sampleCount = 1;
    descriptor.format = wgpu::TextureFormat::RGBA8Unorm;
    descriptor.mipLevelCount = 1;
    descriptor.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
    wgpu::Texture texture = device_.CreateTexture(&descriptor);

    wgpu::ImageCopyTexture destination;
    destination.texture = texture;

    wgpu::TextureDataLayout layout;
    layout.bytesPerRow = stride;
    layout.rowsPerImage = jpeg.height;

    wgpu::Extent3D extent;
    extent.width = jpeg.width;
    extent.height = jpeg.height;

    device_.GetQueue().WriteTexture(&destination, rgba.data(), rgba.size() * sizeof(uint32_t),
                                    &layout, &extent);
    return texture.CreateView();
}
