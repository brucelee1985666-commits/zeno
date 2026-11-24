#include <zeno/core/Graph.h>
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalStatus.h>

#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <nanovdb/io/IO.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

static void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: zeno-neural-volume-node-smoke input.vdb output.znvds output.nvdb\n";
        return 2;
    }

    const fs::path inputPath = fs::absolute(argv[1]);
    const fs::path znvdsPath = fs::absolute(argv[2]);
    const fs::path nvdbPath = fs::absolute(argv[3]);

    try {
        require(fs::is_regular_file(inputPath), "independent input .vdb file does not exist");
        openvdb::initialize();

        auto& session = zeno::getSession();
        require(session.nodeClasses.count("ReadVDB") == 1,
                "Zeno ReadVDB node was not registered in the runtime");
        require(session.nodeClasses.count("ExportNeuralVolumeDataset") == 1,
                "Zeno ExportNeuralVolumeDataset node was not registered in the runtime");
        require(session.nodeClasses.count("ZNVDSFileToNanoVDB") == 1,
                "Zeno ZNVDSFileToNanoVDB node was not registered in the runtime");

        // Execute the actual Zeno graph nodes, not the underlying helper functions.
        auto exportGraph = session.createGraph();
        exportGraph->addNode("ReadVDB", "read_external_vdb");
        exportGraph->setNodeParam("read_external_vdb", "path", inputPath.string());
        exportGraph->setNodeParam("read_external_vdb", "type", std::string("float"));
        exportGraph->addNode("ExportNeuralVolumeDataset", "export_znvds");
        exportGraph->setNodeParam("export_znvds", "path", znvdsPath.string());
        exportGraph->bindNodeInput("export_znvds", "grid", "read_external_vdb", "data");
        exportGraph->applyNodes({"export_znvds"});
        require(!session.globalStatus->failed(),
                "Zeno ExportNeuralVolumeDataset graph execution failed: " +
                session.globalStatus->toJson());
        require(fs::is_regular_file(znvdsPath) && fs::file_size(znvdsPath) > 64,
                "ExportNeuralVolumeDataset node did not produce a non-empty ZNVDS file");

        // Execute the actual Zeno ZNVDSFileToNanoVDB node.
        session.globalStatus->clearState();
        auto convertGraph = session.createGraph();
        convertGraph->addNode("ZNVDSFileToNanoVDB", "convert_znvds");
        convertGraph->setNodeParam("convert_znvds", "inputpath", znvdsPath.string());
        convertGraph->setNodeParam("convert_znvds", "outputpath", nvdbPath.string());
        convertGraph->applyNodes({"convert_znvds"});
        require(!session.globalStatus->failed(),
                "Zeno ZNVDSFileToNanoVDB graph execution failed: " +
                session.globalStatus->toJson());
        require(fs::is_regular_file(nvdbPath) && fs::file_size(nvdbPath) > 64,
                "ZNVDSFileToNanoVDB node did not produce a non-empty NanoVDB file");

        // Independently reopen both source and result from disk and compare active data.
        openvdb::io::File sourceFile(inputPath.string());
        sourceFile.open();
        auto sourceGrids = sourceFile.getGrids();
        sourceFile.close();
        require(sourceGrids && !sourceGrids->empty(), "input .vdb contains no grids");
        auto source = openvdb::gridPtrCast<openvdb::FloatGrid>(sourceGrids->front());
        require(source != nullptr, "first grid in input .vdb is not a FloatGrid");

        auto nanoHandle = nanovdb::io::readGrid<nanovdb::HostBuffer>(nvdbPath.string());
        const auto* nano = nanoHandle.grid<float>();
        require(nano != nullptr, "node output did not reopen as NanoVDB FloatGrid");
        require(std::string(nano->gridName()) == source->getName(),
                "grid name changed through the actual Zeno nodes");
        require(nano->gridClass() == nanovdb::GridClass::FogVolume,
                "grid class changed through the actual Zeno nodes");

        uint64_t activeVoxels = 0;
        uint64_t activeTiles = 0;
        for (auto it = source->tree().cbeginValueOn(); it; ++it) {
            const auto c = it.getCoord();
            const nanovdb::Coord nc(c.x(), c.y(), c.z());
            require(nano->tree().isActive(nc),
                    "an active source voxel/tile is missing from node-generated NanoVDB");
            require(std::abs(nano->tree().getValue(nc) - it.getValue()) < 1e-6f,
                    "an active source value changed through the actual Zeno nodes");
            if (it.getLevel() == 0) ++activeVoxels;
            else ++activeTiles;
        }
        require(activeVoxels > 0, "input fixture did not contain active leaf voxels");
        require(activeTiles > 0, "input fixture did not contain active internal tiles");

        const auto probe = openvdb::Coord(0, 0, 0);
        const auto sourceWorld = source->transform().indexToWorld(openvdb::Vec3d(probe.x(), probe.y(), probe.z()));
        const auto nanoWorld = nano->map().applyMap(nanovdb::Vec3d(probe.x(), probe.y(), probe.z()));
        require(std::abs(sourceWorld.x() - nanoWorld[0]) < 1e-8 &&
                std::abs(sourceWorld.y() - nanoWorld[1]) < 1e-8 &&
                std::abs(sourceWorld.z() - nanoWorld[2]) < 1e-8,
                "index-to-world transform changed through the actual Zeno nodes");

        std::cout << "ZENO NODE GRAPH PASS: ReadVDB -> ExportNeuralVolumeDataset -> "
                  << "ZNVDSFileToNanoVDB; independently reopened .vdb -> .znvds -> .nvdb; "
                  << activeVoxels << " active leaf voxels, " << activeTiles
                  << " active tiles, values/occupancy/metadata/transform verified\n";
        openvdb::uninitialize();
        return 0;
    } catch (const std::exception& e) {
        openvdb::uninitialize();
        std::cerr << "ZENO NODE GRAPH FAIL: " << e.what() << '\n';
        return 1;
    }
}
