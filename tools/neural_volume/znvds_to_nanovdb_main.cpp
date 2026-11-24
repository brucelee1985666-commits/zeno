#include "znvds_to_nanovdb.h"

#include <nanovdb/io/IO.h>

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: znvds_to_nanovdb <input.znvds> <output.nvdb>\n";
        return 2;
    }
    try {
        auto handle = zeno::neural_volume::convertZnvdsToNanoVdb(argv[1]);
        nanovdb::io::writeGrid(std::string(argv[2]), handle);
        std::cout << "Wrote NanoVDB FloatGrid: " << argv[2] << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ZNVDS conversion failed: " << e.what() << "\n";
        return 1;
    }
}
