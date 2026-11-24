cmake_minimum_required(VERSION 3.20)
project(ZenoOpenVDBFloatGridRoundTrip LANGUAGES CXX)

if(NOT DEFINED OPENVDB_SOURCE_DIR OR NOT DEFINED NANOVDB_INCLUDE_DIR)
    message(FATAL_ERROR "Set OPENVDB_SOURCE_DIR and NANOVDB_INCLUDE_DIR")
endif()

# Use OpenVDB's own Find module to link the distro runtime. This avoids
# rebuilding the full library while keeping real FloatGrid execution in CI.
list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_LIST_DIR}"
    "${OPENVDB_SOURCE_DIR}/cmake")
find_package(OpenVDB REQUIRED)

add_executable(openvdb-floatgrid-roundtrip-smoke
    ../openvdb_floatgrid_roundtrip_smoke.cpp
    ../openvdb_floatgrid_to_znvds.cpp
    ../znvds_to_nanovdb.cpp)
target_compile_features(openvdb-floatgrid-roundtrip-smoke PRIVATE cxx_std_17)
target_include_directories(openvdb-floatgrid-roundtrip-smoke PRIVATE
    "${NANOVDB_INCLUDE_DIR}")
target_link_libraries(openvdb-floatgrid-roundtrip-smoke PRIVATE OpenVDB::openvdb)
