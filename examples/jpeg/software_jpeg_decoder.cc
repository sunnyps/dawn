#include "examples/jpeg/software_jpeg_decoder.h"

#include <dawn/webgpu_cpp.h>
#include <setjmp.h>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "jpeglib.h"

struct my_error_mgr {
    struct jpeg_error_mgr pub; /* "public" fields */

    jmp_buf setjmp_buffer; /* for return to caller */
};

typedef struct my_error_mgr* my_error_ptr;

void my_error_exit(j_common_ptr cinfo) {
    my_error_ptr myerr = (my_error_ptr)cinfo->err;
    (*cinfo->err->output_message)(cinfo);
    longjmp(myerr->setjmp_buffer, 1);
}

SoftwareJpegDecoder::SoftwareJpegDecoder(wgpu::Device device)
    : device_(device) {}

SoftwareJpegDecoder::~SoftwareJpegDecoder() = default;

wgpu::TextureView SoftwareJpegDecoder::Decode(std::vector<uint8_t> data, int* width, int* height) {
    my_error_mgr jerr;
    jpeg_decompress_struct cinfo;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        return {};
    }

    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, data.data(), data.size());
    (void)jpeg_read_header(&cinfo, TRUE);
    (void)jpeg_start_decompress(&cinfo);
    int row_stride = cinfo.output_width * cinfo.output_components;

    int buffer_stride = (cinfo.output_width + 63) & 0xffffffc0;
    std::vector<uint32_t> rgba(buffer_stride * cinfo.output_height);

    JSAMPARRAY buffer =
        (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);
    size_t row_offset = 0;
    while (cinfo.output_scanline < cinfo.output_height) {
        (void)jpeg_read_scanlines(&cinfo, buffer, 1);
        for (size_t i = 0, j = 0; i < cinfo.output_width; ++i) {
            uint32_t r = buffer[0][j++];
            uint32_t g = buffer[0][j++];
            uint32_t b = buffer[0][j++];
            rgba[row_offset + i] = r | (g << 8) | (b << 16);
        }
        row_offset += buffer_stride;
    }
    (void)jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

    *width = cinfo.output_width;
    *height = cinfo.output_height;

    wgpu::TextureDescriptor descriptor;
    descriptor.dimension = wgpu::TextureDimension::e2D;
    descriptor.size.width = cinfo.output_width;
    descriptor.size.height = cinfo.output_height;
    descriptor.size.depthOrArrayLayers = 1;
    descriptor.sampleCount = 1;
    descriptor.format = wgpu::TextureFormat::RGBA8Unorm;
    descriptor.mipLevelCount = 1;
    descriptor.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
    wgpu::Texture texture = device_.CreateTexture(&descriptor);

    wgpu::ImageCopyTexture destination;
    destination.texture = texture;

    wgpu::TextureDataLayout layout;
    layout.bytesPerRow = buffer_stride * sizeof(uint32_t);
    layout.rowsPerImage = cinfo.output_height;

    wgpu::Extent3D extent;
    extent.width = cinfo.output_width;
    extent.height = cinfo.output_height;

    device_.GetQueue().WriteTexture(&destination, rgba.data(), rgba.size() * sizeof(uint32_t),
                                    &layout, &extent);
    return texture.CreateView();
}
