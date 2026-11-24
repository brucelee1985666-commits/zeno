#pragma once

#include <openvdb/openvdb.h>

#include <filesystem>

namespace zeno::neural_volume {

// Writes the lossless ZNVDS v1 interchange stream from a real OpenVDB FloatGrid.
// Throws std::runtime_error on unsupported transforms or I/O failures.
void writeZnvdsFromOpenVdbFloatGrid(
    const openvdb::FloatGrid& grid,
    const std::filesystem::path& outputPath);

} // namespace zeno::neural_volume
