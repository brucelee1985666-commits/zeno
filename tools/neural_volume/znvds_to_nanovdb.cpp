#include "znvds_to_nanovdb.h"

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace zeno::neural_volume {
namespace {

class Reader {
public:
    explicit Reader(const std::filesystem::path& path) : in(path, std::ios::binary) {
        if (!in) throw std::runtime_error("cannot open ZNVDS input: " + path.string());
    }
    void exact(char* dst, std::size_t n, const char* label) {
        if (n > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
            throw std::runtime_error(std::string("oversized ZNVDS field: ") + label);
        in.read(dst, static_cast<std::streamsize>(n));
        if (static_cast<std::size_t>(in.gcount()) != n)
            throw std::runtime_error(std::string("truncated ZNVDS ") + label);
    }
    uint16_t u16(const char* label) {
        unsigned char b[2]; exact(reinterpret_cast<char*>(b), 2, label);
        return uint16_t(b[0]) | (uint16_t(b[1]) << 8u);
    }
    uint32_t u32(const char* label) {
        unsigned char b[4]; exact(reinterpret_cast<char*>(b), 4, label);
        return uint32_t(b[0]) | (uint32_t(b[1]) << 8u) |
               (uint32_t(b[2]) << 16u) | (uint32_t(b[3]) << 24u);
    }
    uint64_t u64(const char* label) {
        unsigned char b[8]; exact(reinterpret_cast<char*>(b), 8, label);
        uint64_t v = 0;
        for (unsigned i = 0; i < 8; ++i) v |= uint64_t(b[i]) << (8u * i);
        return v;
    }
    int32_t i32(const char* label) { return static_cast<int32_t>(u32(label)); }
    float f32(const char* label) {
        const uint32_t bits = u32(label);
        float v; std::memcpy(&v, &bits, sizeof(v)); return v;
    }
    double f64(const char* label) {
        const uint64_t bits = u64(label);
        double v; std::memcpy(&v, &bits, sizeof(v)); return v;
    }
    std::string string(const char* label) {
        const uint32_t n = u32(label);
        if (n > 1024u * 1024u) throw std::runtime_error("ZNVDS grid name exceeds 1 MiB");
        std::string s(n, '\0');
        if (n) exact(s.data(), n, label);
        return s;
    }
    bool atEnd() {
        char c;
        if (in.read(&c, 1)) return false;
        if (!in.eof()) throw std::runtime_error("I/O error while checking ZNVDS end");
        return true;
    }
private:
    std::ifstream in;
};

struct Mat3 {
    double a[3][3]{};
    double& operator()(int r, int c) { return a[r][c]; }
    const double& operator()(int r, int c) const { return a[r][c]; }
};

Mat3 inverse(const Mat3& m) {
    const double a=m(0,0), b=m(0,1), c=m(0,2);
    const double d=m(1,0), e=m(1,1), f=m(1,2);
    const double g=m(2,0), h=m(2,1), i=m(2,2);
    const double det = a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);
    if (!std::isfinite(det) || std::abs(det) < 1.0e-15)
        throw std::runtime_error("ZNVDS affine transform is singular");
    Mat3 r;
    r(0,0)=(e*i-f*h)/det; r(0,1)=(c*h-b*i)/det; r(0,2)=(b*f-c*e)/det;
    r(1,0)=(f*g-d*i)/det; r(1,1)=(a*i-c*g)/det; r(1,2)=(c*d-a*f)/det;
    r(2,0)=(d*h-e*g)/det; r(2,1)=(b*g-a*h)/det; r(2,2)=(a*e-b*d)/det;
    return r;
}

void addTile(nanovdb::tools::build::FloatGrid& grid, uint32_t level,
             const nanovdb::Coord& coord, float value) {
    switch (level) {
        case 1: grid.root().template addTile<1>(coord, value, true); break;
        case 2: grid.root().template addTile<2>(coord, value, true); break;
        case 3: grid.root().template addTile<3>(coord, value, true); break;
        default: throw std::runtime_error("ZNVDS tile level must be 1, 2, or 3");
    }
}

} // namespace

nanovdb::GridHandle<nanovdb::HostBuffer>
convertZnvdsToNanoVdb(const std::filesystem::path& path) {
    Reader in(path);
    char magic[8]; in.exact(magic, sizeof(magic), "magic");
    const char expected[8] = {'Z','N','V','D','S','0','0','1'};
    if (std::memcmp(magic, expected, sizeof(magic)) != 0)
        throw std::runtime_error("invalid ZNVDS magic");
    if (in.u32("version") != 1u) throw std::runtime_error("unsupported ZNVDS version");
    if (in.u32("flags") != 3u) throw std::runtime_error("unsupported ZNVDS flags");

    const uint64_t leafCount = in.u64("leaf count");
    const uint64_t tileCount = in.u64("tile count");
    const uint64_t activeCount = in.u64("active voxel count");
    // Conservative limits prevent corrupt counts from causing unbounded allocations/work.
    if (leafCount > 100000000ull || tileCount > 100000000ull ||
        activeCount > leafCount * 512ull)
        throw std::runtime_error("ZNVDS record counts exceed safety limits");

    const std::string name = in.string("grid name");
    const uint32_t gridClass = in.u32("grid class");
    if (gridClass > static_cast<uint32_t>(nanovdb::GridClass::Staggered))
        throw std::runtime_error("unsupported ZNVDS grid class");
    const float background = in.f32("background");
    if (!std::isfinite(background)) throw std::runtime_error("non-finite ZNVDS background");
    for (int axis=0; axis<3; ++axis) {
        const float voxelSize = in.f32("voxel size");
        if (!std::isfinite(voxelSize) || voxelSize <= 0.0f)
            throw std::runtime_error("invalid ZNVDS voxel size");
    }

    double m[4][4]{};
    for (int r=0; r<4; ++r) for (int c=0; c<4; ++c) {
        m[r][c] = in.f64("affine transform");
        if (!std::isfinite(m[r][c])) throw std::runtime_error("non-finite ZNVDS affine transform");
    }
    if (std::abs(m[3][0]) > 1e-12 || std::abs(m[3][1]) > 1e-12 ||
        std::abs(m[3][2]) > 1e-12 || std::abs(m[3][3]-1.0) > 1e-12)
        throw std::runtime_error("ZNVDS transform is not affine");

    Mat3 linear, invLinear;
    for (int r=0; r<3; ++r) for (int c=0; c<3; ++c) linear(r,c)=m[r][c];
    invLinear = inverse(linear);

    nanovdb::tools::build::FloatGrid source(
        background, name, static_cast<nanovdb::GridClass>(gridClass));
    source.mMap.set(linear, invLinear,
                    nanovdb::Vec3d(m[0][3], m[1][3], m[2][3]));

    std::set<std::tuple<int32_t,int32_t,int32_t,uint32_t>> seenTiles;
    for (uint64_t n=0; n<tileCount; ++n) {
        const int32_t x=in.i32("tile x"), y=in.i32("tile y"), z=in.i32("tile z");
        const uint32_t level=in.u32("tile level");
        const float value=in.f32("tile value");
        if (level < 1 || level > 3 || !std::isfinite(value))
            throw std::runtime_error("invalid ZNVDS active tile");
        if (!seenTiles.emplace(x,y,z,level).second)
            throw std::runtime_error("duplicate ZNVDS active tile");
        addTile(source, level, nanovdb::Coord(x,y,z), value);
    }

    std::set<std::tuple<int32_t,int32_t,int32_t>> seenLeaves;
    uint64_t observedActive = 0;
    for (uint64_t leaf=0; leaf<leafCount; ++leaf) {
        const int32_t ox=in.i32("leaf origin x");
        const int32_t oy=in.i32("leaf origin y");
        const int32_t oz=in.i32("leaf origin z");
        if ((ox%8)||(oy%8)||(oz%8))
            throw std::runtime_error("unaligned ZNVDS leaf origin");
        if (!seenLeaves.emplace(ox,oy,oz).second)
            throw std::runtime_error("duplicate ZNVDS leaf origin");
        const uint32_t count=in.u32("leaf active count");
        if (count > 512) throw std::runtime_error("ZNVDS leaf active count exceeds 512");
        std::array<uint8_t,64> mask{};
        in.exact(reinterpret_cast<char*>(mask.data()), mask.size(), "leaf occupancy mask");
        uint32_t bits=0;
        for (uint8_t byte : mask) {
#if defined(__GNUC__) || defined(__clang__)
            bits += static_cast<uint32_t>(__builtin_popcount(static_cast<unsigned>(byte)));
#else
            for (; byte; byte &= static_cast<uint8_t>(byte-1)) ++bits;
#endif
        }
        if (bits != count) throw std::runtime_error("ZNVDS occupancy bit count mismatch");
        std::set<uint16_t> offsets;
        for (uint32_t n=0; n<count; ++n) {
            const uint16_t offset=in.u16("active voxel offset");
            const float value=in.f32("active voxel value");
            if (offset >= 512 || !std::isfinite(value))
                throw std::runtime_error("invalid ZNVDS active voxel");
            if (!offsets.insert(offset).second)
                throw std::runtime_error("duplicate ZNVDS active voxel offset");
            if (!(mask[offset >> 3u] & (1u << (offset & 7u))))
                throw std::runtime_error("ZNVDS value does not match active occupancy bit");
            const int32_t x=ox+(offset >> 6u);
            const int32_t y=oy+((offset >> 3u)&7u);
            const int32_t z=oz+(offset&7u);
            source.tree().setValue(nanovdb::Coord(x,y,z), value);
        }
        observedActive += count;
    }
    if (observedActive != activeCount)
        throw std::runtime_error("ZNVDS active voxel total mismatch");
    if (!in.atEnd()) throw std::runtime_error("unexpected trailing bytes in ZNVDS file");

    return nanovdb::tools::createNanoGrid(source);
}

} // namespace zeno::neural_volume
