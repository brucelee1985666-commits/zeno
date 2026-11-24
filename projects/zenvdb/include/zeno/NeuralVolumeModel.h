#pragma once

#include <array>
#include <cstdint>
#include <istream>
#include <string>
#include <utility>
#include <vector>

namespace zeno::neuralvolume {

struct ActiveTile {
    std::array<int32_t, 3> coord{};
    uint32_t level = 0;
    float value = 0.0f;
};

struct LeafCorrections {
    std::array<int32_t, 3> origin{};
    std::vector<uint16_t> occupancyOffsets;
    std::vector<std::pair<uint16_t, float>> exactValueOffsets;
};

struct EncodedDomain {
    std::array<int32_t, 3> key{};
    std::array<int32_t, 3> origin{};
    std::vector<float> classifierParams;
    bool hasRegressor = false;
    float valueCenter = 0.0f;
    float valueScale = 1.0f;
    std::vector<float> regressorParams;
    std::array<float, 5> metrics{};
    uint32_t occupancyCorrectionCount = 0;
    uint32_t valueCorrectionCount = 0;
    std::vector<LeafCorrections> leaves;
};

struct NeuralVolumeModel {
    uint32_t version = 0;
    uint32_t flags = 0;
    bool partial = false;
    std::array<uint8_t, 32> sourceDatasetSha256{};
    std::string gridName;
    uint32_t gridClass = 0;
    float background = 0.0f;
    std::array<float, 3> voxelSize{};
    std::array<double, 16> indexToWorld{};
    uint32_t domainSizeVoxels = 0;
    uint32_t trainingSteps = 0;
    uint32_t batchSize = 0;
    float learningRate = 0.0f;
    float valueTolerance = 0.0f;
    std::string modelConfigJson;
    uint64_t sourceLeafBlocks = 0;
    uint64_t sourceActiveVoxels = 0;
    std::vector<ActiveTile> activeTiles;
    std::vector<EncodedDomain> domains;
};

void applyLeafCorrections(const LeafCorrections& corrections,
                          std::array<uint8_t, 512>& occupancy,
                          std::array<float, 512>& values);

NeuralVolumeModel readZNVM(std::istream& input);
NeuralVolumeModel readZNVMFile(const std::string& path);

} // namespace zeno::neuralvolume
