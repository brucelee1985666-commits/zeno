#include "znvds_to_nanovdb.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void u16(std::ostream& o, uint16_t v) { o.put(char(v)); o.put(char(v>>8)); }
void u32(std::ostream& o, uint32_t v) { for(int i=0;i<4;++i)o.put(char(v>>(8*i))); }
void u64(std::ostream& o, uint64_t v) { for(int i=0;i<8;++i)o.put(char(v>>(8*i))); }
void i32(std::ostream& o, int32_t v) { u32(o, static_cast<uint32_t>(v)); }
void f32(std::ostream& o, float v) { uint32_t b; std::memcpy(&b,&v,4); u32(o,b); }
void f64(std::ostream& o, double v) { uint64_t b; std::memcpy(&b,&v,8); u64(o,b); }
void writeFixture(const std::filesystem::path& p) {
    std::ofstream o(p,std::ios::binary);
    o.write("ZNVDS001",8);
    u32(o,1); u32(o,3); u64(o,1); u64(o,3); u64(o,2);
    const std::string name="znvds-roundtrip"; u32(o,static_cast<uint32_t>(name.size())); o.write(name.data(),name.size());
    u32(o,2); f32(o,0.0f); f32(o,0.5f); f32(o,1.0f); f32(o,1.5f);
    // Non-uniform scale plus translation, row-major index-to-world affine.
    const double m[16]={2,0.5,0,0, 0,3,0.25,0, 0,0,4,0, 10,-4,7,1};
    for(double v:m) f64(o,v);
    // Active tiles at all three NanoVDB hierarchy levels.
    i32(o,64); i32(o,0); i32(o,0); u32(o,1); f32(o,2.5f);
    i32(o,0); i32(o,128); i32(o,0); u32(o,2); f32(o,3.5f);
    i32(o,4096); i32(o,0); i32(o,0); u32(o,3); f32(o,4.5f);
    i32(o,-8); i32(o,16); i32(o,24); u32(o,2);
    uint8_t mask[64]{}; mask[0]=0x03;
    o.write(reinterpret_cast<const char*>(mask),64);
    u16(o,0); f32(o,0.25f);
    u16(o,1); f32(o,0.75f);
    if(!o) throw std::runtime_error("failed to write ZNVDS fixture");
}
}

int main() {
    const auto path=std::filesystem::temp_directory_path()/"znvds-to-nanovdb-roundtrip.znvds";
    try {
        writeFixture(path);
        auto handle=zeno::neural_volume::convertZnvdsToNanoVdb(path);
        const auto* grid=handle.grid<float>();
        if(!grid) throw std::runtime_error("converter returned no NanoVDB FloatGrid");
        const nanovdb::Coord a(-8,16,24), b(-7,16,24), tileProbe(65,1,1);
        if(!grid->tree().isActive(a)||!grid->tree().isActive(b))
            throw std::runtime_error("round-trip lost active leaf voxels");
        if(std::abs(grid->tree().getValue(a)-0.25f)>1e-6f ||
           std::abs(grid->tree().getValue(b)-0.75f)>1e-6f)
            throw std::runtime_error("round-trip changed leaf values");
        if(!grid->tree().isActive(tileProbe) ||
           std::abs(grid->tree().getValue(tileProbe)-2.5f)>1e-6f)
            throw std::runtime_error("round-trip lost level-1 active tile");
        if(!grid->tree().isActive(nanovdb::Coord(1,129,1)) ||
           std::abs(grid->tree().getValue(nanovdb::Coord(1,129,1))-3.5f)>1e-6f)
            throw std::runtime_error("round-trip lost level-2 active tile");
        if(!grid->tree().isActive(nanovdb::Coord(4097,1,1)) ||
           std::abs(grid->tree().getValue(nanovdb::Coord(4097,1,1))-4.5f)>1e-6f)
            throw std::runtime_error("round-trip lost level-3 active tile");
        if(std::abs(grid->tree().getValue(nanovdb::Coord(100000,100000,100000)))>1e-6f)
            throw std::runtime_error("round-trip changed grid background");
        if(grid->gridClass()!=nanovdb::GridClass::FogVolume)
            throw std::runtime_error("round-trip changed grid class");
        if(std::string(grid->gridName())!="znvds-roundtrip")
            throw std::runtime_error("round-trip changed grid name");
        const auto world=grid->map().applyMap(nanovdb::Vec3d(-8,16,24));
        if(std::abs(world[0]+6.0)>1e-9 || std::abs(world[1]-40.0)>1e-9 ||
           std::abs(world[2]-107.0)>1e-9)
            throw std::runtime_error("round-trip changed affine transform");
        // The C++ parser must reject a truncated file, not silently emit a partial grid.
        const auto badPath=std::filesystem::temp_directory_path()/"znvds-to-nanovdb-truncated.znvds";
        std::filesystem::copy_file(path,badPath,std::filesystem::copy_options::overwrite_existing);
        const auto bytes=std::filesystem::file_size(badPath);
        std::filesystem::resize_file(badPath,bytes-1);
        bool rejected=false;
        try { (void)zeno::neural_volume::convertZnvdsToNanoVdb(badPath); }
        catch (const std::exception&) { rejected=true; }
        std::filesystem::remove(badPath);
        if(!rejected) throw std::runtime_error("converter accepted a truncated ZNVDS file");
        std::filesystem::remove(path);
        std::cout<<"ZNVDS -> NanoVDB CPU round-trip passed: values, occupancy, all tile levels, class, name, affine transform\n";
        return 0;
    } catch (...) {
        std::error_code ec; std::filesystem::remove(path,ec); throw;
    }
}
