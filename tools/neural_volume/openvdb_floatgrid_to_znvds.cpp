#include "openvdb_floatgrid_to_znvds.h"

#include <openvdb/openvdb.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace zeno::neural_volume {
namespace {

void writeU16LE(std::ostream& out, uint16_t value) {
    const char bytes[2] = {static_cast<char>(value & 0xffu),
                           static_cast<char>((value >> 8u) & 0xffu)};
    out.write(bytes, sizeof(bytes));
}
void writeU32LE(std::ostream& out, uint32_t value) {
    const char bytes[4] = {static_cast<char>(value & 0xffu),
                           static_cast<char>((value >> 8u) & 0xffu),
                           static_cast<char>((value >> 16u) & 0xffu),
                           static_cast<char>((value >> 24u) & 0xffu)};
    out.write(bytes, sizeof(bytes));
}
void writeU64LE(std::ostream& out, uint64_t value) {
    for (unsigned i = 0; i != 8; ++i) {
        const char byte = static_cast<char>((value >> (i * 8u)) & 0xffu);
        out.write(&byte, 1);
    }
}
void writeI32LE(std::ostream& out, int32_t value) {
    writeU32LE(out, static_cast<uint32_t>(value));
}
void writeF32LE(std::ostream& out, float value) {
    static_assert(sizeof(float) == sizeof(uint32_t), "ZNVDS requires 32-bit float");
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    writeU32LE(out, bits);
}
void writeF64LE(std::ostream& out, double value) {
    static_assert(sizeof(double) == sizeof(uint64_t), "ZNVDS requires 64-bit double");
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    writeU64LE(out, bits);
}
void writeString(std::ostream& out, const std::string& value) {
    if (value.size() > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("OpenVDB grid name is too long for ZNVDS");
    writeU32LE(out, static_cast<uint32_t>(value.size()));
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

} // namespace

void writeZnvdsFromOpenVdbFloatGrid(
    const openvdb::FloatGrid& grid,
    const std::filesystem::path& outputPath) {
    const auto& transform = grid.transform();
    if (!transform.baseMap()->isLinear())
        throw std::runtime_error("ZNVDS v1 requires a linear/affine OpenVDB transform");
    const auto affineMap = transform.baseMap()->getAffineMap();
    if (!affineMap)
        throw std::runtime_error("could not extract OpenVDB affine transform");

    const auto& tree = grid.tree();
    uint64_t leafCount = 0, tileCount = 0, activeVoxelCount = 0;
    for (auto leafIt = tree.cbeginLeaf(); leafIt; ++leafIt) {
        ++leafCount;
        for (openvdb::Index n = 0; n < leafIt->NUM_VALUES; ++n)
            if (leafIt->isValueOn(n)) ++activeVoxelCount;
    }
    for (auto valueIt = tree.cbeginValueOn(); valueIt; ++valueIt)
        if (valueIt.getLevel() > 0) ++tileCount;

    if (!outputPath.parent_path().empty())
        std::filesystem::create_directories(outputPath.parent_path());
    std::ofstream out(outputPath, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("could not open ZNVDS output: " + outputPath.string());

    const char magic[8] = {'Z','N','V','D','S','0','0','1'};
    out.write(magic, sizeof(magic));
    writeU32LE(out, 1u);
    writeU32LE(out, 3u);
    writeU64LE(out, leafCount);
    writeU64LE(out, tileCount);
    writeU64LE(out, activeVoxelCount);
    writeString(out, grid.getName());
    writeU32LE(out, static_cast<uint32_t>(grid.getGridClass()));
    writeF32LE(out, grid.background());
    const auto voxelSize = grid.voxelSize();
    for (int axis = 0; axis < 3; ++axis)
        writeF32LE(out, static_cast<float>(voxelSize[axis]));
    const auto matrix = affineMap->getMat4();
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            writeF64LE(out, matrix(row, column));

    for (auto valueIt = tree.cbeginValueOn(); valueIt; ++valueIt) {
        if (valueIt.getLevel() <= 0) continue;
        const auto coord = valueIt.getCoord();
        writeI32LE(out, coord.x());
        writeI32LE(out, coord.y());
        writeI32LE(out, coord.z());
        writeU32LE(out, static_cast<uint32_t>(valueIt.getLevel()));
        writeF32LE(out, valueIt.getValue());
    }

    for (auto leafIt = tree.cbeginLeaf(); leafIt; ++leafIt) {
        const auto& leaf = *leafIt;
        const auto origin = leaf.origin();
        writeI32LE(out, origin.x());
        writeI32LE(out, origin.y());
        writeI32LE(out, origin.z());
        uint32_t activeInLeaf = 0;
        for (openvdb::Index n = 0; n < leaf.NUM_VALUES; ++n)
            if (leaf.isValueOn(n)) ++activeInLeaf;
        writeU32LE(out, activeInLeaf);

        std::array<uint8_t, 64> occupancy{};
        for (openvdb::Index n = 0; n < leaf.NUM_VALUES; ++n) {
            if (leaf.isValueOn(n))
                occupancy[n >> 3u] |= static_cast<uint8_t>(1u << (n & 7u));
        }
        out.write(reinterpret_cast<const char*>(occupancy.data()),
                  static_cast<std::streamsize>(occupancy.size()));
        for (openvdb::Index n = 0; n < leaf.NUM_VALUES; ++n) {
            if (!leaf.isValueOn(n)) continue;
            writeU16LE(out, static_cast<uint16_t>(n));
            writeF32LE(out, leaf.getValue(n));
        }
    }

    out.flush();
    if (!out) throw std::runtime_error("failed while writing ZNVDS output: " + outputPath.string());
}

} // namespace zeno::neural_volume
