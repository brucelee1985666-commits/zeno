#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <filesystem>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: write-independent-vdb-fixture output.vdb\n";
        return 2;
    }
    try {
        openvdb::initialize();
        auto grid = openvdb::FloatGrid::create(0.0f);
        grid->setName("independent-node-input");
        grid->setGridClass(openvdb::GRID_FOG_VOLUME);
        openvdb::Mat4R matrix;
        matrix.setIdentity();
        matrix(0,0)=1.5; matrix(0,1)=0.25;
        matrix(1,1)=2.0; matrix(1,2)=0.125;
        matrix(2,2)=2.5;
        matrix(3,0)=3.0; matrix(3,1)=-2.0; matrix(3,2)=5.0;
        grid->setTransform(openvdb::math::Transform::createLinearTransform(matrix));
        grid->tree().setValueOn(openvdb::Coord(-8,16,24), 0.25f);
        grid->tree().setValueOn(openvdb::Coord(-7,16,24), 0.75f);
        grid->tree().setValueOn(openvdb::Coord(8,16,24), 1.0f);
        grid->tree().addTile(1, openvdb::Coord(64,0,0), 2.5f, true);
        grid->tree().addTile(2, openvdb::Coord(0,128,0), 3.5f, true);
        grid->tree().addTile(3, openvdb::Coord(4096,0,0), 4.5f, true);
        openvdb::io::File(argv[1]).write({grid});
        openvdb::uninitialize();
        std::cout << "Independent .vdb fixture written: " << argv[1] << '\n';
        return 0;
    } catch (const std::exception& e) {
        openvdb::uninitialize();
        std::cerr << e.what() << '\n';
        return 1;
    }
}
