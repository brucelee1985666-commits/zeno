#include <zeno/NeuralVolumeModel.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

void u16(std::ostream& out, uint16_t value) {
    const char b[2] = {char(value & 0xff), char((value >> 8) & 0xff)};
    out.write(b, 2);
}
void u32(std::ostream& out, uint32_t value) {
    const char b[4] = {char(value), char(value >> 8), char(value >> 16), char(value >> 24)};
    out.write(b, 4);
}
void u64(std::ostream& out, uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) out.put(char(value >> (i * 8)));
}
void i32(std::ostream& out, int32_t value) { u32(out, static_cast<uint32_t>(value)); }
void f32(std::ostream& out, float value) {
    uint32_t bits = 0; std::memcpy(&bits, &value, 4); u32(out, bits);
}
void f64(std::ostream& out, double value) {
    uint64_t bits = 0; std::memcpy(&bits, &value, 8); u64(out, bits);
}
void string(std::ostream& out, const std::string& value) {
    u32(out, static_cast<uint32_t>(value.size())); out.write(value.data(), value.size());
}

std::string fixture(bool partial, bool trailing, int corruption = 0) {
    std::ostringstream out(std::ios::out | std::ios::binary);
    out.write("ZNVM0001", 8);
    u32(out, 1); u32(out, partial ? 7 : 3);
    char hash[32]{}; out.write(hash, sizeof(hash));
    string(out, "test");
    u32(out, 2); f32(out, 0.0f);
    f32(out, 1.0f); f32(out, 1.0f); f32(out, 1.0f);
    const double matrix[16] = {
        1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1
    };
    for (double v : matrix) f64(out, v);
    u32(out, 128); u32(out, 2000); u32(out, 32768);
    f32(out, 0.01f); f32(out, 0.001f);
    string(out, "{\"encoding\":{\"otype\":\"HashGrid\"}}");
    u64(out, 0); // tiles
    u64(out, 1); // domains
    u64(out, partial ? 2 : 1); // source leaves
    u64(out, 2); // active leaf voxels

    i32(out, 0); i32(out, 0); i32(out, 0); // domain key
    i32(out, 0); i32(out, 0); i32(out, 0); // domain origin
    u64(out, 1); f32(out, corruption == 4 ? std::numeric_limits<float>::quiet_NaN() : 0.5f); // classifier params
    u32(out, 0); // no regressor
    f32(out, 1); f32(out, 1); f32(out, 1); f32(out, 1); f32(out, -1);
    u32(out, 1); u32(out, 1); // occupancy/value correction totals
    u32(out, 1); // leaf count
    i32(out, 0); i32(out, 0); i32(out, 0);
    u32(out, corruption == 1 ? 2 : 1); u16(out, 7); // occupancy correction
    if (corruption == 1) u16(out, 7);
    u32(out, corruption == 2 ? 2 : 1); u16(out, 3); f32(out, corruption == 3 ? std::numeric_limits<float>::quiet_NaN() : 0.75f); // exact value correction
    if (corruption == 2) { u16(out, 3); f32(out, 0.5f); }
    if (trailing) out.put('x');
    return out.str();
}

} // namespace

int main() {
    {
        std::istringstream input(fixture(false, false), std::ios::binary);
        auto model = zeno::neuralvolume::readZNVM(input);
        assert(model.gridName == "test");
        assert(model.domains.size() == 1);
        assert(model.domains[0].leaves.size() == 1);
        assert(model.domains[0].leaves[0].occupancyOffsets[0] == 7);
        assert(model.domains[0].leaves[0].exactValueOffsets[0].first == 3);
        assert(!model.partial);
    }
    {
        zeno::neuralvolume::LeafCorrections corrections;
        corrections.occupancyOffsets = {7};
        corrections.exactValueOffsets = {{3, 0.75f}};
        std::array<uint8_t, 512> occupancy{};
        std::array<float, 512> values{};
        occupancy[7] = 0;
        values[3] = 0.1f;
        zeno::neuralvolume::applyLeafCorrections(corrections, occupancy, values);
        assert(occupancy[7] == 1);
        assert(values[3] == 0.75f);
    }
    {
        std::istringstream input(fixture(true, false), std::ios::binary);
        auto model = zeno::neuralvolume::readZNVM(input);
        assert(model.partial);
        assert(model.sourceLeafBlocks == 2);
    }
    {
        std::istringstream input(fixture(false, true), std::ios::binary);
        bool rejected = false;
        try { (void)zeno::neuralvolume::readZNVM(input); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
    }
    for (int corruption : {1, 2, 3, 4}) {
        std::istringstream input(fixture(false, false, corruption), std::ios::binary);
        bool rejected = false;
        try { (void)zeno::neuralvolume::readZNVM(input); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
    }
    return 0;
}
