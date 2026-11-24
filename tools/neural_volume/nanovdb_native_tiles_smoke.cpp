// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <nanovdb/tools/CreateNanoGrid.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

int main() {
    using BuildT = nanovdb::ValueOnIndex;
    using GridT = nanovdb::tools::build::Grid<BuildT>;

    GridT source(1u, "native-active-tile-smoke");
    const std::array<nanovdb::Coord, 4> probes{
        nanovdb::Coord(0, 0, 0),       // active leaf voxel
        nanovdb::Coord(64, 0, 0),      // level-1 tile
        nanovdb::Coord(256, 0, 0),     // level-2 tile
        nanovdb::Coord(1024, 0, 0)     // level-3/root tile
    };

    source.setValue(probes[0], 1u);
    source.root().template addTile<1>(probes[1], 1u, true);
    source.root().template addTile<2>(probes[2], 1u, true);
    source.root().template addTile<3>(probes[3], 1u, true);

    auto hostHandle = nanovdb::tools::createNanoGrid(source, 0u, false, true, 0);
    const auto* indexGrid = hostHandle.grid<BuildT>();
    if (!indexGrid) throw std::runtime_error("CreateNanoGrid returned no ValueOnIndex grid");

    auto accessor = indexGrid->getAccessor();
    std::array<uint32_t, probes.size()> indices{};
    for (size_t i = 0; i < probes.size(); ++i) {
        indices[i] = accessor.getValue(probes[i]);
        if (indices[i] == 0) {
            throw std::runtime_error("active leaf or tile was omitted from the ValueOnIndex topology");
        }
        for (size_t j = 0; j < i; ++j) {
            if (indices[i] == indices[j]) {
                throw std::runtime_error("active leaf and tile values did not receive unique sidecar indices");
            }
        }
    }

    std::cout << "Native NanoVDB tile topology preserved; indexed values:";
    for (const auto index : indices) std::cout << " " << index;
    std::cout << "\n";
    return 0;
}
