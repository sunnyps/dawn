#ifndef JPEG_DATA_H_
#define JPEG_DATA_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

class DataView {
  public:
    explicit DataView(const std::vector<uint8_t>& data);
    DataView(const std::vector<uint8_t>& data, size_t offset, size_t size);

    size_t bytes_remaining() const {
        return size_ - (offset_ - base_offset_);
    }
    size_t size() const {
        return size_;
    }

    std::string AsString() const;
    void Advance(size_t size);
    DataView Slice(size_t size) const;
    uint8_t ReadByte();
    void ReadBytes(uint8_t* dest, size_t num_bytes);
    uint16_t ReadUint16();

  private:
    const std::vector<uint8_t>& data_;
    const size_t base_offset_;
    const size_t size_;
    size_t offset_;
};

class ScanDataStream {
  public:
    explicit ScanDataStream(DataView& view);

    uint8_t ReadBit();
    uint32_t ReadBits(size_t num_bits);

  private:
    DataView& view_;
    uint8_t current_byte_ = 0;
    size_t bit_offset_ = 8;
};

class HuffmanTable {
  public:
    HuffmanTable(uint8_t lengths[16], const std::vector<uint8_t>& elements);

    uint8_t GetCode(ScanDataStream& stream) const;

  private:
    struct Node {
        bool is_leaf = false;
        uint8_t value = 0;
        std::vector<Node> children;
    };

    static bool Build(Node& root, uint8_t value, size_t code_length);

    Node root_;
};

struct ScanComponentInfo {
    uint8_t id;
    uint8_t quantization_table_index;
    uint8_t dc_table_index;
    uint8_t ac_table_index;
};

struct ScanData {
    std::vector<ScanComponentInfo> components;
    std::vector<uint8_t> coded_data;
};

using IntBlock = std::array<int32_t, 64>;
using IntBlockList = std::vector<IntBlock>;
using IntBlockMap = std::map<uint8_t, IntBlockList>;

using ByteBlock = std::array<uint8_t, 64>;
using ByteBlockList = std::vector<ByteBlock>;
using ByteBlockMap = std::map<uint8_t, ByteBlockList>;

struct JpegData {
    uint32_t width;
    uint32_t height;
    std::map<uint8_t, HuffmanTable> huffman_tables;
    std::map<uint8_t, ByteBlock> quantization_tables;
    std::vector<ScanData> scan_data_blocks;
};

bool ParseJpegData(std::vector<uint8_t> raw_data, JpegData& jpeg);

///// Helpers for different phases of decoding which may be replaceable with GPU work.

// Produces all 8x8 DCT blocks in raster order from the Huffman-coded scan data in `jpeg`. Each
// map entry corresponds to a single component of the image.
IntBlockMap DecodeDCTBlocks(const JpegData& jpeg);

// Rearranges the contents of a set of 8x8 DCT blocks to each be in zig-zag order, and performs an
// inverse DCT to recover blocks of color components.
IntBlockMap PerformIDCT(const IntBlockMap& dct_blocks);

std::vector<uint32_t> ConvertYCbCrToRGBA(const JpegData& jpeg,
                                         const IntBlockMap& blocks,
                                         uint32_t* stride);

#endif  // JPEG_DATA_H_
