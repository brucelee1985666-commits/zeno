#include "openvdb_floatgrid_to_znvds.h"
#include "znvds_to_nanovdb.h"

#include <openvdb/openvdb.h>
#include <nanovdb/io/IO.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

openvdb::FloatGrid::Ptr makeRealFloatGrid() {
    auto grid = openvdb::FloatGrid::create(0.0f);
    grid->setName("openvdb-real-floatgrid");
    grid->setGridClass(openvdb::GRID_FOG_VOLUME);

    openvdb::Mat4R matrix;
    matrix.setIdentity();
    matrix(0,0)=2.0; matrix(0,1)=0.5;
    matrix(1,1)=3.0; matrix(1,2)=0.25;
    matrix(2,2)=4.0;
    matrix(3,0)=10.0; matrix(3,1)=-4.0; matrix(3,2)=7.0;
    grid->setTransform(openvdb::math::Transform::createLinearTransform(matrix));

    grid->tree().setValueOn(openvdb::Coord(-8,16,24), 0.25f);
    grid->tree().setValueOn(openvdb::Coord(-7,16,24), 0.75f);
    grid->tree().addTile(1, openvdb::Coord(64,0,0), 2.5f, true);
    grid->tree().addTile(2, openvdb::Coord(0,128,0), 3.5f, true);
    grid->tree().addTile(3, openvdb::Coord(4096,0,0), 4.5f, true);
    return grid;
}
}

int main() {
    openvdb::initialize();
    const auto base = std::filesystem::temp_directory_path();
    const auto znvdsPath = base / "openvdb-floatgrid-real-chain.znvds";
    const auto nvdbPath = base / "openvdb-floatgrid-real-chain.nvdb";
    try {
        // This is a genuine OpenVDB FloatGrid tree, not a hand-authored ZNVDS fixture.
        auto source = makeRealFloatGrid();
        zeno::neural_volume::writeZnvdsFromOpenVdbFloatGrid(*source, znvdsPath);
        require(std::filesystem::exists(znvdsPath) &&
                std::filesystem::file_size(znvdsPath) > 64,
                "OpenVDB FloatGrid exporter did not produce ZNVDS bytes");

        auto converted = zeno::neural_volume::convertZnvdsToNanoVdb(znvdsPath);
        nanovdb::io::writeGrid(nvdbPath.string(), converted);
        auto reopened = nanovdb::io::readGrid<nanovdb::HostBuffer>(nvdbPath.string());
        const auto* result = reopened.grid<float>();
        require(result != nullptr, "serialized result did not reopen as NanoVDB FloatGrid");

        require(std::string(result->gridName()) == source->getName(),
                "grid name changed across OpenVDB -> ZNVDS -> NanoVDB");
        require(result->gridClass() == nanovdb::GridClass::FogVolume,
                "grid class changed across conversion");
        require(std::abs(result->tree().getValue(nanovdb::Coord(100000,100000,100000))) < 1e-6f,
                "background changed across conversion");

        uint64_t checkedLeafVoxels = 0;
        uint64_t checkedTiles = 0;
        for (auto it = source->tree().cbeginValueOn(); it; ++it) {
            const auto c = it.getCoord();
            const nanovdb::Coord nc(c.x(), c.y(), c.z());
            require(result->tree().isActive(nc),
                    "an active OpenVDB voxel/tile was lost in NanoVDB");
            require(std::abs(result->tree().getValue(nc) - it.getValue()) < 1e-6f,
                    "an active OpenVDB value changed in NanoVDB");
            if (it.getLevel() == 0) ++checkedLeafVoxels;
            else ++checkedTiles;
        }
        require(checkedLeafVoxels == 2, "unexpected OpenVDB active leaf voxel count");
        require(checkedTiles == 3, "unexpected OpenVDB active tile count");

        const auto world = result->map().applyMap(nanovdb::Vec3d(-8,16,24));
        const auto sourceWorld = source->transform().indexToWorld(openvdb::Vec3d(-8,16,24));
        require(std::abs(world[0]-sourceWorld.x()) < 1e-9 &&
                std::abs(world[1]-sourceWorld.y()) < 1e-9 &&
                std::abs(world[2]-sourceWorld.z()) < 1e-9,
                "affine index-to-world transform changed across conversion");

        std::filesystem::remove(znvdsPath);
        std::filesystem::remove(nvdbPath);
        std::cout << "REAL OpenVDB FloatGrid -> ZNVDS -> NanoVDB disk round-trip passed; "
                  << checkedLeafVoxels << " active voxels, " << checkedTiles
                  << " active tiles, metadata and affine transform verified\n";
        openvdb::uninitialize();
        return 0;
    } catch (...) {
        std::error_code ec;
        std::filesystem::remove(znvdsPath, ec);
        std::filesystem::remove(nvdbPath, ec);
        openvdb::uninitialize();
        throw;
    }
}
