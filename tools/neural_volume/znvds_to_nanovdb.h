#pragma once

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

#include <cstdint>
#include <filesystem>

namespace zeno::neural_volume {

// Converts the lossless ZNVDS v1 interchange stream into a host NanoVDB FloatGrid.
// Throws std::runtime_error on malformed or unsupported input.
nanovdb::GridHandle<nanovdb::HostBuffer>
convertZnvdsToNanoVdb(const std::filesystem::path& path);

} // namespace zeno::neural_volume
