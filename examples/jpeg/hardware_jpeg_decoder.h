#ifndef HARDWARE_JPEG_DECODER_H_
#define HARDWARE_JPEG_DECODER_H_

#include <dawn/webgpu_cpp.h>

#include "examples/jpeg/jpeg_decoder.h"

class HardwareJpegDecoder : public JpegDecoder {
 public:
  explicit HardwareJpegDecoder(wgpu::Device device);
  ~HardwareJpegDecoder() override;

  // JpegDecoder:
  wgpu::TextureView Decode(std::vector<uint8_t> data, int* width, int* height) override;

 private:
  wgpu::Device device_;
  wgpu::ComputePipeline pipeline_;
};

#endif  // HARDWARE_JPEG_DECODER_H_
