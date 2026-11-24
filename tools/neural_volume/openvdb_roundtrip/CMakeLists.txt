cmake_minimum_required(VERSION 3.20)
project(ZenoOpenVDBFloatGridRoundTrip LANGUAGES CXX)

if(NOT DEFINED NANOVDB_INCLUDE_DIR)
    message(FATAL_ERROR "Set NANOVDB_INCLUDE_DIR")
endif()

# Link the distro OpenVDB runtime for a fast executable CPU test. The production
# Zeno exporter is separately syntax-checked against the pinned OpenVDB submodule.
find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenVDB REQUIRED IMPORTED_TARGET OpenVDB)

add_executable(openvdb-floatgrid-roundtrip-smoke
    ../openvdb_floatgrid_roundtrip_smoke.cpp
    ../openvdb_floatgrid_to_znvds.cpp
    ../znvds_to_nanovdb.cpp)
target_compile_features(openvdb-floatgrid-roundtrip-smoke PRIVATE cxx_std_17)
target_include_directories(openvdb-floatgrid-roundtrip-smoke PRIVATE
    "${NANOVDB_INCLUDE_DIR}")
target_link_libraries(openvdb-floatgrid-roundtrip-smoke PRIVATE PkgConfig::OpenVDB)
