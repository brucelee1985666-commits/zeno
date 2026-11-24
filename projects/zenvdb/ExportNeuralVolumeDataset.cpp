#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>

#include "openvdb_floatgrid_to_znvds.h"

namespace zeno {

struct ExportNeuralVolumeDataset : zeno::INode {
    void apply() override {
        auto input = get_input<VDBGrid>("grid");
        auto grid = std::dynamic_pointer_cast<VDBFloatGrid>(input);
        if (!grid || !grid->m_grid)
            throw zeno::Exception(
                "ExportNeuralVolumeDataset currently accepts FloatGrid only");

        const std::string path = get_param<std::string>("path");
        if (path.empty())
            throw zeno::Exception("ExportNeuralVolumeDataset requires an output path");

        try {
            neural_volume::writeZnvdsFromOpenVdbFloatGrid(*grid->m_grid, path);
        } catch (const std::exception& error) {
            throw zeno::Exception(
                std::string("ExportNeuralVolumeDataset failed: ") + error.what());
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

} // namespace zeno
