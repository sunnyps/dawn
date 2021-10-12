#ifndef SOFTWARE_JPEG_DECODER_H_
#define SOFTWARE_JPEG_DECODER_H_

#include <dawn/webgpu_cpp.h>

#include "examples/jpeg/jpeg_decoder.h"

class SoftwareJpegDecoder : public JpegDecoder {
 public:
  explicit SoftwareJpegDecoder(wgpu::Device device);
  ~SoftwareJpegDecoder() override;

  // JpegDecoder:
  wgpu::TextureView Decode(std::vector<uint8_t> data, int* width, int* height) override;

 private:
  wgpu::Device device_;
};

#endif  // SOFTWARE_JPEG_DECODER_H_
