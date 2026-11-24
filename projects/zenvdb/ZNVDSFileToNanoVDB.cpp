#include <zeno/zeno.h>

#include "znvds_to_nanovdb.h"

#include <nanovdb/io/IO.h>

#include <filesystem>
#include <string>

namespace zeno {
namespace {

struct ZNVDSFileToNanoVDB : INode {
    void apply() override {
        const auto inputPath = get_param<std::string>("inputpath");
        const auto outputPath = get_param<std::string>("outputpath");
        if (inputPath.empty()) {
            throw zeno::Exception("ZNVDSFileToNanoVDB requires an input .znvds path");
        }
        if (outputPath.empty()) {
            throw zeno::Exception("ZNVDSFileToNanoVDB requires an output .nvdb path");
        }

        const std::filesystem::path destination(outputPath);
        if (!destination.parent_path().empty()) {
            std::filesystem::create_directories(destination.parent_path());
        }

        try {
            auto handle = zeno::neural_volume::convertZnvdsToNanoVdb(inputPath);
            if (!handle || !handle.grid<float>()) {
                throw zeno::Exception("ZNVDSFileToNanoVDB conversion produced no FloatGrid");
            }
            nanovdb::io::writeGrid(
                destination.string(),
                handle,
                nanovdb::io::Codec::NONE,
                0);
        } catch (const zeno::Exception&) {
            throw;
        } catch (const std::exception& e) {
            throw zeno::Exception(
                std::string("ZNVDSFileToNanoVDB failed: ") + e.what());
        }
    }
};

ZENDEFNODE(ZNVDSFileToNanoVDB,
    {
        /* inputs: */ {
        },
        /* outputs: */ {
        },
        /* params: */ {
            {"readpath", "inputpath", ""},
            {"writepath", "outputpath", ""},
        },
        /* category: */ {
            "nanovdb",
        }
    });

} // namespace
} // namespace zeno
