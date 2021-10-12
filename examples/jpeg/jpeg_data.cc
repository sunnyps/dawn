#include <cassert>
#include <cmath>
#include <iostream>
#include <map>

#include <fcntl.h>
#include <unistd.h>

#include "examples/jpeg/jpeg_data.h"

constexpr uint16_t kMarker_StartImage = 0xffd8;
constexpr uint16_t kMarker_EndImage = 0xffd9;

constexpr uint16_t kMarker_DefineHuffmanTables = 0xffc4;
constexpr uint16_t kMarker_DefineQuantizationTables = 0xffdb;
constexpr uint16_t kMarker_DefineRestartInterval = 0xffdd;

constexpr uint16_t kMarker_StartOfFrame_BaselineDCT = 0xffc0;
constexpr uint16_t kMarker_StartOfFrame_ProgressiveDCT = 0xffc2;

constexpr uint16_t kMarker_StartOfScan = 0xffda;

constexpr uint16_t kMarker_AppDefaultHeader = 0xffe0;

constexpr uint16_t kMarker_Comment = 0xfffe;

DataView::DataView(const std::vector<uint8_t>& data) : DataView(data, 0, data.size()) {
}

DataView::DataView(const std::vector<uint8_t>& data, size_t offset, size_t size)
    : data_(data), base_offset_(offset), size_(size), offset_(base_offset_) {
}

std::string DataView::AsString() const {
    return std::string(data_.data() + offset_, data_.data() + offset_ + size_);
}

void DataView::Advance(size_t size) {
    assert(bytes_remaining() >= size);
    offset_ += size;
}

DataView DataView::Slice(size_t size) const {
    return DataView(data_, offset_, size);
}

uint8_t DataView::ReadByte() {
    assert(bytes_remaining() >= 1);
    return data_[offset_++];
}

void DataView::ReadBytes(uint8_t* dest, size_t num_bytes) {
    assert(bytes_remaining() >= num_bytes);
    std::copy(data_.data() + offset_, data_.data() + offset_ + num_bytes, dest);
    Advance(num_bytes);
}

uint16_t DataView::ReadUint16() {
    assert(bytes_remaining() >= 2);
    uint16_t value = (static_cast<uint16_t>(data_[offset_]) << 8) | data_[offset_ + 1];
    offset_ += 2;
    return value;
}

struct FrameComponentInfo {
    uint8_t horizontal_sampling_factor;
    uint8_t vertical_sampling_factor;
    uint8_t quantization_table_index;
    uint16_t width;
    uint16_t height;
};

struct FrameInfo {
    uint8_t precision;
    uint16_t max_width;
    uint16_t max_height;
    uint8_t max_horizontal_sampling_factor = 0;
    uint8_t max_vertical_sampling_factor = 0;
    std::map<uint8_t, FrameComponentInfo> components;
};

bool ReadHuffmanTables(DataView& view, JpegData& jpeg) {
    while (view.bytes_remaining()) {
        uint8_t id = view.ReadByte();

        if (view.bytes_remaining() < 16) {
            return false;
        }

        uint8_t lengths[16];
        view.ReadBytes(lengths, 16);
        std::vector<uint8_t> elements;
        for (uint8_t num_elements : lengths) {
            if (num_elements == 0) {
                continue;
            }
            size_t index = elements.size();
            elements.resize(elements.size() + num_elements);
            if (view.bytes_remaining() < num_elements) {
                return false;
            }
            view.ReadBytes(&elements[index], num_elements);
        }

        jpeg.huffman_tables.emplace(id, HuffmanTable(lengths, elements));
    }
    return true;
}

bool ReadQuantizationTables(DataView& view, JpegData& jpeg) {
    while (view.bytes_remaining() >= 65) {
        uint8_t id = view.ReadByte();
        view.ReadBytes(reinterpret_cast<uint8_t*>(jpeg.quantization_tables[id].data()), 64);
    }
    return true;
}

bool ReadBaselineDCT(DataView& view, FrameInfo& frame) {
    if (view.size() < 6) {
        std::cerr << "bad header\n";
        return false;
    }

    frame.precision = view.ReadByte();
    frame.max_height = view.ReadUint16();
    frame.max_width = view.ReadUint16();

    uint8_t num_components = view.ReadByte();
    if (view.size() < 6 + 3 * num_components) {
        return false;
    }

    for (size_t i = 0; i < num_components; ++i) {
        FrameComponentInfo& component = frame.components[view.ReadByte()];
        uint8_t sampling_factors = view.ReadByte();
        component.horizontal_sampling_factor = sampling_factors >> 4;
        component.vertical_sampling_factor = sampling_factors & 0xf;
        component.quantization_table_index = view.ReadByte();

        if (component.horizontal_sampling_factor > frame.max_horizontal_sampling_factor) {
            frame.max_horizontal_sampling_factor = component.horizontal_sampling_factor;
        }
        if (component.vertical_sampling_factor > frame.max_vertical_sampling_factor) {
            frame.max_vertical_sampling_factor = component.vertical_sampling_factor;
        }
    }

    for (auto& entry : frame.components) {
        FrameComponentInfo& component = entry.second;
        component.width = (frame.max_width * component.horizontal_sampling_factor) /
                          frame.max_horizontal_sampling_factor;
        component.height = (frame.max_height * component.vertical_sampling_factor) /
                           frame.max_vertical_sampling_factor;
    }

    return true;
}

bool ReadScanHeader(DataView& view, const FrameInfo& frame, ScanData& scan) {
    if (view.size() < 4) {
        return false;
    }

    uint8_t num_components = view.ReadByte();
    if (num_components == 0 || view.size() < 4 + num_components * 2) {
        return false;
    }

    scan.components.resize(num_components);
    for (ScanComponentInfo& component : scan.components) {
        component.id = view.ReadByte();
        auto it = frame.components.find(component.id);
        if (it == frame.components.end()) {
            return false;
        }

        component.quantization_table_index = it->second.quantization_table_index;

        uint8_t coding_table_indices = view.ReadByte();
        component.dc_table_index = coding_table_indices >> 4;
        component.ac_table_index = coding_table_indices & 0xf;
    }

    // Ignore some fields we don't care about...
    view.ReadByte();
    view.ReadByte();
    view.ReadByte();
    return true;
}

size_t ComputeScanDataLength(const DataView& view) {
    DataView slice = view.Slice(view.size());
    size_t length = 0;
    while (slice.bytes_remaining()) {
        if (slice.ReadByte() == 0xff) {
            if (slice.bytes_remaining() == 0 || slice.ReadByte() != 0) {
                return length;
            }
            length += 2;
        } else {
            ++length;
        }
    }
    return length;
}

ScanDataStream::ScanDataStream(DataView& view) : view_(view) {
}

uint8_t ScanDataStream::ReadBit() {
    if (bit_offset_ == 8) {
        current_byte_ = view_.ReadByte();
        if (current_byte_ == 0xff) {
            assert(view_.bytes_remaining() >= 1);
            uint8_t zero = view_.ReadByte();
            (void)zero;
            assert(zero == 0);
        }
        bit_offset_ = 0;
    }

    uint8_t bit = (current_byte_ >> (7 - bit_offset_)) & 1;
    ++bit_offset_;
    return bit;
}

uint32_t ScanDataStream::ReadBits(size_t num_bits) {
    assert(num_bits <= 32);
    uint32_t value = 0;
    while (num_bits--) {
        value = (value << 1) | ReadBit();
    }
    return value;
}

HuffmanTable::HuffmanTable(uint8_t lengths[16], const std::vector<uint8_t>& elements) {
    auto element_it = elements.begin();
    for (size_t i = 0; i < 16; ++i) {
        for (size_t j = 0; j < lengths[i]; ++j) {
            Build(root_, *element_it++, i + 1);
        }
    }
}

uint8_t HuffmanTable::GetCode(ScanDataStream& stream) const {
    const Node* node = &root_;
    while (!node->is_leaf) {
        uint8_t bit = stream.ReadBit();
        assert(bit <= 1);
        if (bit >= node->children.size()) {
            std::cerr << "missing huffman code\n";
            return 0;
        }
        node = &node->children[bit];
    }
    return node->value;
}

// static
bool HuffmanTable::Build(Node& node, uint8_t value, size_t code_length) {
    if (node.is_leaf)
        return false;

    if (code_length == 0) {
        node.is_leaf = true;
        node.value = value;
        return true;
    }

    for (size_t i = 0; i < 2; ++i) {
        if (i >= node.children.size()) {
            node.children.emplace_back();
        }
        if (Build(node.children.back(), value, code_length - 1)) {
            return true;
        }
    }

    return false;
}

class JpegLoader {
  public:
    explicit JpegLoader(std::vector<uint8_t> raw_data)
        : data_(std::move(raw_data)), view_(data_) {}

    bool Load(JpegData& jpeg) {
        FrameInfo frame;
        while (view_.bytes_remaining() >= 2) {
            const uint16_t marker = view_.ReadUint16();
            if (marker == kMarker_StartImage) {
                continue;
            }
            if (marker == kMarker_EndImage) {
                break;
            }

            if (view_.bytes_remaining() < 2) {
                return false;
            }

            uint16_t segment_length = view_.ReadUint16();
            if (segment_length < 2) {
                std::cerr << "Invalid segment length\n";
                return false;
            }

            segment_length -= 2;
            if (view_.bytes_remaining() < segment_length) {
                std::cerr << "Segment runs past EOF\n";
                return false;
            }

            DataView segment_view = view_.Slice(segment_length);
            view_.Advance(segment_length);

            switch (marker) {
                case kMarker_DefineHuffmanTables:
                    if (!ReadHuffmanTables(segment_view, jpeg)) {
                        return false;
                    }
                    break;

                case kMarker_DefineQuantizationTables:
                    if (!ReadQuantizationTables(segment_view, jpeg)) {
                        return false;
                    }
                    break;

                case kMarker_StartOfFrame_BaselineDCT:
                    if (!frame.components.empty()) {
                        std::cerr << "Duplicate frame data?\n";
                        return false;
                    }
                    if (!ReadBaselineDCT(segment_view, frame)) {
                        return false;
                    }
                    break;

                case kMarker_StartOfScan: {
                    if (frame.components.empty()) {
                        std::cerr << "Scan data without a frame header?\n";
                        return false;
                    }

                    ScanData scan;
                    if (!ReadScanHeader(segment_view, frame, scan)) {
                        return false;
                    }

                    size_t scan_data_length = ComputeScanDataLength(view_);
                    scan.coded_data.resize(scan_data_length);
                    view_.ReadBytes(scan.coded_data.data(), scan_data_length);
                    jpeg.scan_data_blocks.push_back(std::move(scan));
                    break;
                }

                case kMarker_DefineRestartInterval:
                    std::cout << "Ignoring restart interval definition\n";
                    break;

                case kMarker_AppDefaultHeader:
                    std::cout << "Ignoring application default header\n";
                    break;

                case kMarker_StartOfFrame_ProgressiveDCT:
                    std::cerr << "Rejecting progressive encoding\n";
                    return false;

                case kMarker_Comment:
                    break;

                default:
                    break;
            }
        }

        jpeg.width = frame.max_width;
        jpeg.height = frame.max_height;

        return true;
    }

  private:
    const std::vector<uint8_t> data_;
    DataView view_;
};

bool ParseJpegData(std::vector<uint8_t> raw_data, JpegData& jpeg) {
    return JpegLoader(std::move(raw_data)).Load(jpeg);
}

int32_t DecodeNumber(uint8_t code, uint32_t bits) {
    int32_t value = 1 << (code - 1);
    if (static_cast<int32_t>(bits) >= value) {
        return bits;
    }
    return static_cast<int32_t>(bits) - (value << 1) + 1;
}

const uint8_t kZigZagTable[] = {
    0,  1,  5,  6,  14, 15, 27, 28, 2,  4,  7,  13, 16, 26, 29, 42, 3,  8,  12, 17, 25, 30,
    41, 43, 9,  11, 18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38,
    46, 51, 55, 60, 21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63,
};

float NormCoeff(float n) {
    if (n == 0) {
        return 1.0f / sqrtf(2.0f);
    } else {
        return 1.0f;
    }
}

const float* GetIDCTTable() {
    static float table[64];
    static bool initializer([&] {
        for (size_t u = 0; u < 8; ++u) {
            for (size_t x = 0; x < 8; ++x) {
                const size_t index = x + u * 8;
                table[index] = NormCoeff(u) * cosf(((2.0f * x + 1.0f) * u * 3.14159f) / 16.0f);
            }
        }
        return true;
    }());
    (void)initializer;
    return table;
}

IntBlockMap DecodeDCTBlocks(const JpegData& jpeg) {
    IntBlockMap all_blocks;
    std::map<uint8_t, int> dc_coefficients;
    for (const ScanData& scan : jpeg.scan_data_blocks) {
        DataView view(scan.coded_data);
        ScanDataStream stream(view);

        while (view.bytes_remaining()) {
            for (const ScanComponentInfo& component : scan.components) {
                const uint8_t id = component.id;
                IntBlockList& component_blocks = all_blocks[id];
                component_blocks.emplace_back();
                IntBlock& new_block = component_blocks.back();

                auto quant_it = jpeg.quantization_tables.find(component.quantization_table_index);
                if (quant_it == jpeg.quantization_tables.end()) {
                    return {};
                }

                const ByteBlock& quantization_table = quant_it->second;

                auto huff_it = jpeg.huffman_tables.find(component.dc_table_index);
                if (huff_it == jpeg.huffman_tables.end()) {
                    return {};
                }

                uint8_t code = huff_it->second.GetCode(stream);
                uint32_t bits = stream.ReadBits(code);
                int dc_coefficient = (dc_coefficients[id] += DecodeNumber(code, bits));
                new_block[0] = dc_coefficient * quantization_table[0];
                size_t j = 1;
                while (j < 64) {
                    huff_it = jpeg.huffman_tables.find(component.ac_table_index | 0x10);
                    if (huff_it == jpeg.huffman_tables.end()) {
                        return {};
                    }

                    code = huff_it->second.GetCode(stream);
                    if (code == 0) {
                        break;
                    }

                    if (code & 0xf0) {
                        j += code >> 4;
                        code = code & 0xf;
                    }
                    bits = stream.ReadBits(code);
                    if (j < 64) {
                        new_block[j] = quantization_table[j] * DecodeNumber(code, bits);
                        ++j;
                    }
                }
            }
        }
    }

    return all_blocks;
}

IntBlockMap PerformIDCT(const IntBlockMap& dct_blocks) {
    IntBlockMap output;
    for (const auto& entry : dct_blocks) {
        const uint8_t component_id = entry.first;
        const IntBlockList& in_blocks = entry.second;
        IntBlockList& out_blocks = output[component_id];
        out_blocks.resize(in_blocks.size());

        IntBlockList::iterator out_block_it = out_blocks.begin();
        for (const IntBlock& in_block : in_blocks) {
            // First do the zig-zag.
            IntBlock zigzag;
            for (size_t i = 0; i < 64; ++i) {
                zigzag[i] = in_block[kZigZagTable[i]];
            }

            // Now do the IDCT
            IntBlock& out_block = *out_block_it++;
            for (size_t y = 0; y < 8; ++y) {
                for (size_t x = 0; x < 8; ++x) {
                    float local_sum = 0;
                    for (size_t v = 0; v < 8; ++v) {
                        for (size_t u = 0; u < 8; ++u) {
                            local_sum += static_cast<float>(zigzag[u + v * 8]) *
                                         GetIDCTTable()[x + u * 8] * GetIDCTTable()[y + v * 8];
                        }
                    }

                    out_block[x + y * 8] = static_cast<int>(local_sum / 4.0f);
                }
            }
        }
    }
    return output;
}

// For convenience when working with decoded texture data on the GPU, we round buffer stride up to
// multiples of 64 RGBA (32-bit) values, or 256 bytes.
uint32_t RoundTo64(uint32_t n) {
    return (n + 63) & 0xffffffc0;
}

uint32_t AdjustAndClamp(float x) {
    x += 128;
    if (x > 255) {
        return 255;
    }
    if (x < 0) {
        return 0;
    }
    return static_cast<uint32_t>(x);
}

uint32_t YCbCrToRGB(int y, int cb, int cr) {
    float r = static_cast<float>(cr) * (2.0f - 2.0f * 0.299f) + static_cast<float>(y);
    float b = static_cast<float>(cb) * (2.0f - 2.0f * 0.114f) + static_cast<float>(y);
    float g = (static_cast<float>(y) - 0.114f * b - 0.299f * r) / 0.587f;
    return AdjustAndClamp(r) | (AdjustAndClamp(g) << 8) | (AdjustAndClamp(b) << 16) | 0xff000000;
}

std::vector<uint32_t> ConvertYCbCrToRGBA(const JpegData& jpeg,
                                         const IntBlockMap& blocks,
                                         uint32_t* out_stride) {
    const uint32_t stride = RoundTo64(jpeg.width);
    std::vector<uint32_t> rgba(stride * jpeg.height);

    if (blocks.empty()) {
        return {};
    }

    IntBlockList::const_iterator y_blocks_it;
    IntBlockList::const_iterator cr_blocks_it;
    IntBlockList::const_iterator cb_blocks_it;

    auto y_blocks_entry = blocks.find(1);
    auto cb_blocks_entry = blocks.find(2);
    auto cr_blocks_entry = blocks.find(3);
    if (y_blocks_entry != blocks.end()) {
        y_blocks_it = y_blocks_entry->second.begin();
    }
    if (cb_blocks_entry != blocks.end()) {
        cb_blocks_it = cb_blocks_entry->second.begin();
    }
    if (cr_blocks_entry != blocks.end()) {
        cr_blocks_it = cr_blocks_entry->second.begin();
    }

    for (size_t block_y = 0; block_y < jpeg.height / 8; ++block_y) {
        for (size_t block_x = 0; block_x < jpeg.width / 8; ++block_x) {
            const IntBlock* y_block = nullptr;
            const IntBlock* cb_block = nullptr;
            const IntBlock* cr_block = nullptr;

            if (y_blocks_entry != blocks.end()) {
                y_block = &(*y_blocks_it++);
            }
            if (cb_blocks_entry != blocks.end()) {
                cb_block = &(*cb_blocks_it++);
            }
            if (cr_blocks_entry != blocks.end()) {
                cr_block = &(*cr_blocks_it++);
            }

            for (size_t v = 0; v < 8; ++v) {
                for (size_t u = 0; u < 8; ++u) {
                    size_t index = u + v * 8;
                    int y = y_block ? (*y_block)[index] : 0;
                    int cb = cb_block ? (*cb_block)[index] : 0;
                    int cr = cr_block ? (*cr_block)[index] : 0;
                    rgba[block_y * 8 * stride + block_x * 8 + v * stride + u] =
                        YCbCrToRGB(y, cb, cr);
                }
            }
        }
    }

    *out_stride = stride * sizeof(uint32_t);
    return rgba;
}
