// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <nanovdb/tools/CreateNanoGrid.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
    using GridT = nanovdb::tools::build::FloatGrid;
    const nanovdb::Coord a(-1, 2, 3);
    const nanovdb::Coord b(0, 2, 3);
    const nanovdb::Coord tileOrigin(8192, 0, 0);
    const nanovdb::Coord tileProbe(8193, 1, 1);

    GridT source(0.0f, "cpu-reference-smoke");
    source.setValue(a, 0.25f);
    source.setValue(b, 0.75f);
    source.root().template addTile<1>(tileOrigin, 2.5f, true);

    auto handle = nanovdb::tools::createNanoGrid(source, 0u, false, true, 0);
    const auto* grid = handle.grid<float>();
    if (!grid) throw std::runtime_error("CPU NanoVDB builder returned no FloatGrid");

    if (!grid->isActive(a) || !grid->isActive(b)) {
        throw std::runtime_error("CPU NanoVDB conversion lost active leaf voxels");
    }
    if (std::abs(grid->getValue(a) - 0.25f) > 1.0e-6f ||
        std::abs(grid->getValue(b) - 0.75f) > 1.0e-6f) {
        throw std::runtime_error("CPU NanoVDB conversion changed leaf scalar values");
    }
    if (!grid->isActive(tileProbe) ||
        std::abs(grid->getValue(tileProbe) - 2.5f) > 1.0e-6f) {
        throw std::runtime_error("CPU NanoVDB conversion did not preserve active tile value");
    }

    std::cout << "CPU NanoVDB FloatGrid reference passed: active leaf values and active tile preserved\n";
    return 0;
}
