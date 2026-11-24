#include <openvdb/openvdb.h>
#include <nanovdb/util/OpenToNanoVDB.h>
#include <nanovdb/util/NanoToOpenVDB.h>
#include <nanovdb/util/IO.h>

#include <stdexcept>

// Compile-only API check against the exact OpenVDB/NanoVDB submodule pinned by Zeno.
// This intentionally does not claim that the complete Zeno target has been built.
void zenoNanoVDBApiCompileSmoke(const openvdb::FloatGrid& source)
{
    auto handle = nanovdb::openToNanoVDB(source);
    if (!handle.grid<float>()) {
        throw std::runtime_error("NanoVDB FloatGrid conversion failed");
    }

    nanovdb::io::writeGrid("compile-smoke.nvdb", handle, nanovdb::io::Codec::NONE, 0);
    auto readHandle = nanovdb::io::readGrid("compile-smoke.nvdb");
    if (!readHandle.grid<float>()) {
        throw std::runtime_error("NanoVDB file read failed");
    }

    auto baseGrid = nanovdb::nanoToOpenVDB(readHandle);
    auto roundTrip = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
    if (!roundTrip) {
        throw std::runtime_error("NanoVDB FloatGrid round-trip failed");
    }

}
