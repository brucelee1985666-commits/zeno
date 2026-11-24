cmake_minimum_required(VERSION 3.24)
project(ZenoOpenVDBFloatGridRoundTrip LANGUAGES CXX)

if(NOT DEFINED OPENVDB_SOURCE_DIR OR NOT DEFINED NANOVDB_INCLUDE_DIR)
    message(FATAL_ERROR "Set OPENVDB_SOURCE_DIR and NANOVDB_INCLUDE_DIR")
endif()

set(OPENVDB_BUILD_BINARIES OFF CACHE BOOL "" FORCE)
set(OPENVDB_BUILD_UNITTESTS OFF CACHE BOOL "" FORCE)
set(OPENVDB_BUILD_PYTHON_MODULE OFF CACHE BOOL "" FORCE)
set(OPENVDB_BUILD_NANOVDB OFF CACHE BOOL "" FORCE)
set(OPENVDB_BUILD_CORE ON CACHE BOOL "" FORCE)
set(USE_NANOVDB OFF CACHE BOOL "" FORCE)
set(USE_EXR OFF CACHE BOOL "" FORCE)
set(USE_IMATH_HALF OFF CACHE BOOL "" FORCE)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

add_subdirectory("${OPENVDB_SOURCE_DIR}" openvdb-build)

add_executable(openvdb-floatgrid-roundtrip-smoke
    ../openvdb_floatgrid_roundtrip_smoke.cpp
    ../openvdb_floatgrid_to_znvds.cpp
    ../znvds_to_nanovdb.cpp)
target_compile_features(openvdb-floatgrid-roundtrip-smoke PRIVATE cxx_std_17)
target_include_directories(openvdb-floatgrid-roundtrip-smoke PRIVATE
    "${NANOVDB_INCLUDE_DIR}")
target_link_libraries(openvdb-floatgrid-roundtrip-smoke PRIVATE openvdb)
