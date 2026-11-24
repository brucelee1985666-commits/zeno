#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

namespace zeno {
namespace {

void writeU16LE(std::ostream& out, uint16_t value) {
    const char bytes[2] = {
        static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8u) & 0xffu)
    };
    out.write(bytes, sizeof(bytes));
}

void writeU32LE(std::ostream& out, uint32_t value) {
    const char bytes[4] = {
        static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8u) & 0xffu),
        static_cast<char>((value >> 16u) & 0xffu),
        static_cast<char>((value >> 24u) & 0xffu)
    };
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
    if (value.size() > std::numeric_limits<uint32_t>::max()) {
        throw zeno::Exception("Neural volume grid name is too long");
    }
    writeU32LE(out, static_cast<uint32_t>(value.size()));
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

struct ExportNeuralVolumeDataset : zeno::INode {
    void apply() override {
        auto input = get_input<VDBGrid>("grid");
        auto grid = std::dynamic_pointer_cast<VDBFloatGrid>(input);
        if (!grid || !grid->m_grid) {
            throw zeno::Exception(
                "ExportNeuralVolumeDataset currently accepts FloatGrid only");
        }

        const std::string path = get_param<std::string>("path");
        if (path.empty()) {
            throw zeno::Exception("ExportNeuralVolumeDataset requires an output path");
        }

        const auto outputPath = std::filesystem::path(path);
        if (!outputPath.parent_path().empty()) {
            std::filesystem::create_directories(outputPath.parent_path());
        }

        const auto& transform = grid->m_grid->transform();
        if (!transform.baseMap()->isLinear()) {
            throw zeno::Exception(
                "ExportNeuralVolumeDataset currently requires a linear/affine grid transform");
        }
        const auto affineMap = transform.baseMap()->getAffineMap();
        if (!affineMap) {
            throw zeno::Exception("Could not extract affine transform for neural volume dataset");
        }

        const auto& tree = grid->m_grid->tree();
        uint64_t leafCount = 0;
        uint64_t tileCount = 0;
        uint64_t activeVoxelCount = 0;

        // Count first so the file header can be written without buffering the
        // entire volume or seeking through an incomplete variable-length table.
        for (auto leafIt = tree.cbeginLeaf(); leafIt; ++leafIt) {
            const auto& leaf = *leafIt;
            ++leafCount;
            for (openvdb::Index n = 0; n < leaf.NUM_VALUES; ++n) {
                if (leaf.isValueOn(n)) ++activeVoxelCount;
            }
        }
        for (auto valueIt = tree.cbeginValueOn(); valueIt; ++valueIt) {
            if (valueIt.getLevel() > 0) ++tileCount;
        }

        std::ofstream out(outputPath, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw zeno::Exception("Could not open neural volume dataset for writing: " + path);
        }

        // ZNVDS v1 is a training-data interchange format, not the final
        // compressed neural-volume container.
        const char magic[8] = {'Z','N','V','D','S','0','0','1'};
        out.write(magic, sizeof(magic));
        writeU32LE(out, 1u); // format version
        writeU32LE(out, 3u); // flags: FloatGrid + affine transform; coordinates are index-space by format
        writeU64LE(out, leafCount);
        writeU64LE(out, tileCount);
        writeU64LE(out, activeVoxelCount);
        writeString(out, grid->m_grid->getName());
        writeU32LE(out, static_cast<uint32_t>(grid->m_grid->getGridClass()));
        writeF32LE(out, grid->m_grid->background());
        const auto voxelSize = grid->m_grid->voxelSize();
        writeF32LE(out, static_cast<float>(voxelSize[0]));
        writeF32LE(out, static_cast<float>(voxelSize[1]));
        writeF32LE(out, static_cast<float>(voxelSize[2]));
        const auto matrix = affineMap->getMat4();
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 4; ++column) {
                writeF64LE(out, matrix(row, column));
            }
        }

        // Preserve active internal tiles explicitly. The neural model will
        // learn only the lower leaf-level topology and values in this first
        // dataset version.
        for (auto valueIt = tree.cbeginValueOn(); valueIt; ++valueIt) {
            if (valueIt.getLevel() <= 0) continue;
            const auto coord = valueIt.getCoord();
            writeI32LE(out, coord.x());
            writeI32LE(out, coord.y());
            writeI32LE(out, coord.z());
            writeU32LE(out, static_cast<uint32_t>(valueIt.getLevel()));
            writeF32LE(out, valueIt.getValue());
        }

        // Each VDB leaf is 8^3. Store its exact 512-bit occupancy mask and
        // active values indexed by their local linear offset. This gives the
        // future classifier and regressor separate, lossless training targets.
        for (auto leafIt = tree.cbeginLeaf(); leafIt; ++leafIt) {
            const auto& leaf = *leafIt;
            const auto origin = leaf.origin();
            writeI32LE(out, origin.x());
            writeI32LE(out, origin.y());
            writeI32LE(out, origin.z());

            uint32_t activeInLeaf = 0;
            for (openvdb::Index n = 0; n < leaf.NUM_VALUES; ++n) {
                if (leaf.isValueOn(n)) ++activeInLeaf;
            }
            writeU32LE(out, activeInLeaf);

            std::array<uint8_t, 64> occupancy{};
            for (openvdb::Index n = 0; n < leaf.NUM_VALUES; ++n) {
                if (leaf.isValueOn(n)) {
                    occupancy[n >> 3u] |= static_cast<uint8_t>(1u << (n & 7u));
                }
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
        if (!out) {
            throw zeno::Exception("Failed while writing neural volume dataset: " + path);
        }
    }
};

static int defExportNeuralVolumeDataset =
    zeno::defNodeClass<ExportNeuralVolumeDataset>(
        "ExportNeuralVolumeDataset",
        {
            /* inputs: */ {"grid"},
            /* outputs: */ {},
            /* params: */ {{"writepath", "path", ""}},
            /* category: */ {"nanovdb"}
        });

} // namespace
} // namespace zeno
