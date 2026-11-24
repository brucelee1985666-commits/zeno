#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>

#include <nanovdb/util/OpenToNanoVDB.h>
#include <nanovdb/util/IO.h>

#include <filesystem>
#include <memory>
#include <string>

namespace zeno {
namespace {

struct OpenVDBToNanoVDB : INode {
    void apply() override {
        auto input = get_input<VDBGrid>("grid");
        auto floatGrid = std::dynamic_pointer_cast<VDBFloatGrid>(input);
        if (!floatGrid || !floatGrid->m_grid) {
            throw zeno::Exception(
                "OpenVDBToNanoVDB currently accepts a non-empty FloatGrid only");
        }

        const auto path = get_param<std::string>("path");
        if (path.empty()) {
            throw zeno::Exception("OpenVDBToNanoVDB requires an output .nvdb path");
        }

        const std::filesystem::path outputPath(path);
        const auto parent = outputPath.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }

        // Use the NanoVDB converter already vendored by the pinned OpenVDB
        // submodule. This first integration deliberately keeps full precision;
        // neural/quantized value coding is a separate codec stage.
        auto handle = nanovdb::openToNanoVDB(*floatGrid->m_grid);
        if (!handle || !handle.grid<float>()) {
            throw zeno::Exception("OpenVDBToNanoVDB conversion produced an empty grid");
        }

        nanovdb::io::writeGrid(
            outputPath.string(),
            handle,
            nanovdb::io::Codec::NONE,
            0);
    }
};

ZENDEFNODE(OpenVDBToNanoVDB,
    {
        /* inputs: */ {
            "grid",
        },
        /* outputs: */ {
        },
        /* params: */ {
            {"writepath", "path", ""},
        },
        /* category: */ {
            "nanovdb",
        }
    });

} // namespace
} // namespace zeno
