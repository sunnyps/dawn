#ifndef SOFTWARE_DECODE_JPEG_H_
#define SOFTWARE_DECODE_JPEG_H_

#include <dawn/webgpu_cpp.h>

#include "examples/jpeg/jpeg_data.h"

wgpu::TextureView SoftwareDecodeJpeg(wgpu::Device device, const JpegData& jpeg);

#endif  // SOFTWARE_DECODE_JPEG_H_
