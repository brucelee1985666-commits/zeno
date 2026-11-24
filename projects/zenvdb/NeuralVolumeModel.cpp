#include <zeno/NeuralVolumeModel.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace zeno::neuralvolume {
namespace {

constexpr char kMagic[8] = {'Z', 'N', 'V', 'M', '0', '0', '0', '1'};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kBaseFlags = 3;
constexpr uint32_t kPartialFlag = 4;
constexpr uint64_t kMaxDomains = 1000000;
constexpr uint64_t kMaxLeaves = 10000000;
constexpr uint64_t kMaxTiles = 10000000;
constexpr uint64_t kMaxParams = 100000000;
constexpr uint32_t kMaxStringBytes = 16u * 1024u * 1024u;

void readExact(std::istream& input, char* data, std::streamsize size, const char* label) {
    input.read(data, size);
    if (input.gcount() != size) {
        throw std::runtime_error(std::string("ZNVM truncated while reading ") + label);
    }
}

uint16_t readU16(std::istream& input, const char* label) {
    unsigned char b[2]{};
    readExact(input, reinterpret_cast<char*>(b), 2, label);
    return static_cast<uint16_t>(b[0] | (uint16_t(b[1]) << 8));
}

uint32_t readU32(std::istream& input, const char* label) {
    unsigned char b[4]{};
    readExact(input, reinterpret_cast<char*>(b), 4, label);
    return uint32_t(b[0]) | (uint32_t(b[1]) << 8) |
           (uint32_t(b[2]) << 16) | (uint32_t(b[3]) << 24);
}

uint64_t readU64(std::istream& input, const char* label) {
    unsigned char b[8]{};
    readExact(input, reinterpret_cast<char*>(b), 8, label);
    uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) value |= uint64_t(b[i]) << (i * 8);
    return value;
}

int32_t readI32(std::istream& input, const char* label) {
    return static_cast<int32_t>(readU32(input, label));
}

float readF32(std::istream& input, const char* label) {
    const uint32_t bits = readU32(input, label);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

double readF64(std::istream& input, const char* label) {
    const uint64_t bits = readU64(input, label);
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::string readString(std::istream& input, const char* label) {
    const uint32_t size = readU32(input, label);
    if (size > kMaxStringBytes) {
        throw std::runtime_error(std::string("ZNVM string too large: ") + label);
    }
    std::string value(size, '\0');
    if (size) readExact(input, value.data(), static_cast<std::streamsize>(size), label);
    return value;
}

std::vector<float> readParams(std::istream& input, const char* label) {
    const uint64_t count = readU64(input, label);
    if (count > kMaxParams || count > std::numeric_limits<size_t>::max() / sizeof(float)) {
        throw std::runtime_error(std::string("ZNVM parameter array too large: ") + label);
    }
    std::vector<float> params(static_cast<size_t>(count));
    for (auto& value : params) {
        value = readF32(input, label);
        if (!std::isfinite(value)) {
            throw std::runtime_error(std::string("ZNVM contains non-finite ") + label);
        }
    }
    return params;
}

} // namespace

void applyLeafCorrections(const LeafCorrections& corrections,
                          std::array<uint8_t, 512>& occupancy,
                          std::array<float, 512>& values) {
    for (uint8_t active : occupancy) {
        if (active > 1) {
            throw std::invalid_argument("leaf occupancy prediction must contain only 0 or 1");
        }
    }
    for (uint16_t offset : corrections.occupancyOffsets) {
        if (offset >= occupancy.size()) {
            throw std::out_of_range("leaf occupancy correction offset out of range");
        }
        occupancy[offset] = static_cast<uint8_t>(1u - occupancy[offset]);
    }
    for (const auto& correction : corrections.exactValueOffsets) {
        const uint16_t offset = correction.first;
        const float value = correction.second;
        if (offset >= values.size()) {
            throw std::out_of_range("leaf value correction offset out of range");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("leaf exact-value correction must be finite");
        }
        values[offset] = value;
    }
}

NeuralVolumeModel readZNVM(std::istream& input) {
    char magic[8]{};
    readExact(input, magic, sizeof(magic), "magic");
    if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("invalid ZNVM magic");
    }

    NeuralVolumeModel model;
    model.version = readU32(input, "version");
    model.flags = readU32(input, "flags");
    if (model.version != kVersion ||
        (model.flags != kBaseFlags && model.flags != (kBaseFlags | kPartialFlag))) {
        throw std::runtime_error("unsupported ZNVM version or flags");
    }
    model.partial = (model.flags & kPartialFlag) != 0;

    readExact(input, reinterpret_cast<char*>(model.sourceDatasetSha256.data()),
              static_cast<std::streamsize>(model.sourceDatasetSha256.size()), "source hash");
    model.gridName = readString(input, "grid name");
    model.gridClass = readU32(input, "grid class");
    model.background = readF32(input, "background");
    if (!std::isfinite(model.background)) {
        throw std::runtime_error("ZNVM contains non-finite background value");
    }
    for (auto& value : model.voxelSize) value = readF32(input, "voxel size");
    for (auto& value : model.indexToWorld) value = readF64(input, "affine transform");

    model.domainSizeVoxels = readU32(input, "domain size");
    model.trainingSteps = readU32(input, "training steps");
    model.batchSize = readU32(input, "batch size");
    model.learningRate = readF32(input, "learning rate");
    model.valueTolerance = readF32(input, "value tolerance");
    model.modelConfigJson = readString(input, "model config");

    const uint64_t tileCount = readU64(input, "tile count");
    const uint64_t domainCount = readU64(input, "domain count");
    model.sourceLeafBlocks = readU64(input, "source leaf count");
    model.sourceActiveVoxels = readU64(input, "source active voxel count");
    if (tileCount > kMaxTiles || domainCount > kMaxDomains ||
        model.sourceLeafBlocks > kMaxLeaves) {
        throw std::runtime_error("ZNVM topology count exceeds safety limit");
    }
    if (model.domainSizeVoxels == 0 || model.domainSizeVoxels % 8 != 0 ||
        model.trainingSteps == 0 || model.batchSize == 0 ||
        !std::isfinite(model.learningRate) || model.learningRate <= 0.0f ||
        !std::isfinite(model.valueTolerance) || model.valueTolerance < 0.0f) {
        throw std::runtime_error("ZNVM contains invalid training parameters");
    }
    for (float value : model.voxelSize) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error("ZNVM contains invalid voxel size");
        }
    }
    for (double value : model.indexToWorld) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("ZNVM contains non-finite affine transform");
        }
    }

    model.activeTiles.reserve(static_cast<size_t>(tileCount));
    for (uint64_t i = 0; i < tileCount; ++i) {
        ActiveTile tile;
        for (auto& coord : tile.coord) coord = readI32(input, "tile coordinate");
        tile.level = readU32(input, "tile level");
        tile.value = readF32(input, "tile value");
        if (!std::isfinite(tile.value)) {
            throw std::runtime_error("ZNVM tile has non-finite value");
        }
        if (tile.level == 0) throw std::runtime_error("ZNVM tile has invalid level");
        model.activeTiles.push_back(tile);
    }

    model.domains.reserve(static_cast<size_t>(domainCount));
    uint64_t encodedLeafCount = 0;
    for (uint64_t domainIndex = 0; domainIndex < domainCount; ++domainIndex) {
        EncodedDomain domain;
        for (auto& value : domain.key) value = readI32(input, "domain key");
        for (auto& value : domain.origin) value = readI32(input, "domain origin");
        domain.classifierParams = readParams(input, "classifier parameters");

        const uint32_t hasRegressor = readU32(input, "regressor flag");
        if (hasRegressor > 1) throw std::runtime_error("ZNVM has invalid regressor flag");
        domain.hasRegressor = hasRegressor != 0;
        if (domain.hasRegressor) {
            domain.valueCenter = readF32(input, "regressor value center");
            domain.valueScale = readF32(input, "regressor value scale");
            if (!std::isfinite(domain.valueScale) || domain.valueScale <= 0.0f) {
                throw std::runtime_error("ZNVM has invalid regressor value scale");
            }
            domain.regressorParams = readParams(input, "regressor parameters");
        }

        for (auto& metric : domain.metrics) {
            metric = readF32(input, "domain metric");
            if (!std::isfinite(metric)) {
                throw std::runtime_error("ZNVM domain contains non-finite metric");
            }
        }
        domain.occupancyCorrectionCount = readU32(input, "domain occupancy correction count");
        domain.valueCorrectionCount = readU32(input, "domain value correction count");
        const uint32_t leafCount = readU32(input, "domain leaf count");
        encodedLeafCount += leafCount;
        if (encodedLeafCount > model.sourceLeafBlocks || encodedLeafCount > kMaxLeaves) {
            throw std::runtime_error("ZNVM encoded leaf count exceeds source bounds");
        }

        domain.leaves.reserve(leafCount);
        uint64_t observedOccupancyCorrections = 0;
        uint64_t observedValueCorrections = 0;
        for (uint32_t leafIndex = 0; leafIndex < leafCount; ++leafIndex) {
            LeafCorrections leaf;
            for (auto& value : leaf.origin) value = readI32(input, "leaf origin");
            const uint32_t occupancyCount = readU32(input, "leaf occupancy correction count");
            if (occupancyCount > 512) throw std::runtime_error("ZNVM occupancy corrections exceed leaf size");
            observedOccupancyCorrections += occupancyCount;
            leaf.occupancyOffsets.reserve(occupancyCount);
            std::array<bool, 512> seenOccupancy{};
            for (uint32_t i = 0; i < occupancyCount; ++i) {
                const uint16_t offset = readU16(input, "occupancy correction offset");
                if (offset >= 512) throw std::runtime_error("ZNVM occupancy offset out of range");
                if (seenOccupancy[offset]) throw std::runtime_error("ZNVM contains duplicate occupancy correction offset");
                seenOccupancy[offset] = true;
                leaf.occupancyOffsets.push_back(offset);
            }
            const uint32_t valueCount = readU32(input, "leaf value correction count");
            if (valueCount > 512) throw std::runtime_error("ZNVM value corrections exceed leaf size");
            observedValueCorrections += valueCount;
            leaf.exactValueOffsets.reserve(valueCount);
            std::array<bool, 512> seenValues{};
            for (uint32_t i = 0; i < valueCount; ++i) {
                const uint16_t offset = readU16(input, "value correction offset");
                const float value = readF32(input, "exact value correction");
                if (offset >= 512) throw std::runtime_error("ZNVM value offset out of range");
                if (seenValues[offset]) throw std::runtime_error("ZNVM contains duplicate exact-value correction offset");
                if (!std::isfinite(value)) throw std::runtime_error("ZNVM contains non-finite exact-value correction");
                seenValues[offset] = true;
                leaf.exactValueOffsets.emplace_back(offset, value);
            }
            domain.leaves.push_back(std::move(leaf));
        }
        if (observedOccupancyCorrections != domain.occupancyCorrectionCount ||
            observedValueCorrections != domain.valueCorrectionCount) {
            throw std::runtime_error("ZNVM domain correction counts do not match leaf records");
        }
        model.domains.push_back(std::move(domain));
    }

    if (encodedLeafCount != model.sourceLeafBlocks && !model.partial) {
        throw std::runtime_error("full ZNVM model is missing source leaf blocks");
    }
    if (encodedLeafCount > model.sourceLeafBlocks) {
        throw std::runtime_error("ZNVM contains more leaf blocks than its source");
    }
    if (input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("ZNVM contains trailing bytes");
    }
    return model;
}

NeuralVolumeModel readZNVMFile(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("could not open ZNVM file: " + path);
    return readZNVM(input);
}

} // namespace zeno::neuralvolume
