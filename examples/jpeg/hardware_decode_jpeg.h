#ifndef HARDWARE_DECODE_JPEG_H_
#define HARDWARE_DECODE_JPEG_H_

#include <dawn/webgpu_cpp.h>

#include "examples/jpeg/jpeg_data.h"

wgpu::TextureView HardwareDecodeJpeg(wgpu::Device device, const JpegData& jpeg);

#endif  // HARDWARE_DECODE_JPEG_H_
