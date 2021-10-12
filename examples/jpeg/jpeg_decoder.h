#ifndef JPEG_DECODER_H_
#define JPEG_DECODER_H_

#include <dawn/webgpu_cpp.h>

#include <cstdint>
#include <vector>

class JpegDecoder {
 public:
  virtual ~JpegDecoder() = default;
  virtual wgpu::TextureView Decode(std::vector<uint8_t> data, int* width, int* height) = 0;
};

#endif  // JPEG_DECODER_H_
