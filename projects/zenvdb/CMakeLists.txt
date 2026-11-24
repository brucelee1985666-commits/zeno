file(GLOB PROJECT_SOURCE include/*/*.h *.cpp include/zeno/packed3grids.cpp)

if (UNIX)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/local/lib/cmake/OpenVDB)
endif()

####################################################
# FUN FACT: DID U KNOW THAT MICROSOFT LOVES LINUX? #
####################################################
if (MSVC)  # must use this to make msvc linker happy
    # 节数超过对象文件格式限制: 请使用 /bigobj 进行编译 
    target_compile_options(zeno
        PRIVATE
        $<IF:$<COMPILE_LANGUAGE:CXX>, /bigobj, >
    )
endif()
####################################################
# FUN FACT: DID U KNOW THAT MICROSOFT BUYS GITHUB? #
####################################################

option(ZENO_SYSTEM_OPENVDB "Use OpenVDB found in system instead of submodule" OFF)

target_include_directories(zeno PRIVATE ../oldzenbase/include)

# Never mix NanoVDB headers from the vendored OpenVDB revision with a different
# system OpenVDB ABI. Resolve both from the same installation/source tree.
if (ZENO_SYSTEM_OPENVDB)
    find_path(ZENO_SYSTEM_NANOVDB_INCLUDE_DIR nanovdb/NanoVDB.h)
    set(ZENO_NANOVDB_INCLUDE_DIR ${ZENO_SYSTEM_NANOVDB_INCLUDE_DIR})
else()
    set(ZENO_NANOVDB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/openvdb/nanovdb)
    if (NOT EXISTS "${ZENO_NANOVDB_INCLUDE_DIR}/nanovdb/NanoVDB.h")
        set(ZENO_NANOVDB_INCLUDE_DIR "")
    endif()
endif()

# NanoVDB utility nodes remain optional if matching headers are unavailable.
if (ZENO_NANOVDB_INCLUDE_DIR)
    target_include_directories(zeno PRIVATE ${ZENO_NANOVDB_INCLUDE_DIR})
else()
    list(FILTER PROJECT_SOURCE EXCLUDE REGEX "(OpenVDBToNanoVDB|ExportNeuralVolumeDataset)\\.cpp$")
    message(STATUS "Matching NanoVDB headers not found: disabling Zeno NanoVDB utility nodes")
endif()

# Add sources only after the optional NanoVDB files have been filtered.
target_sources(zeno PRIVATE ${PROJECT_SOURCE})

# The neural inference backend is optional: the standard Zeno build does not
# acquire a new CUDA/tiny-cuda-nn dependency.
option(ZENO_WITH_TINY_CUDA_NN "Enable the optional tiny-cuda-nn neural volume runtime" OFF)
if (ZENO_WITH_TINY_CUDA_NN)
    if (NOT CMAKE_CUDA_COMPILER_LOADED)
        enable_language(CUDA)
    endif()
    if (NOT TARGET tiny-cuda-nn)
        message(FATAL_ERROR
            "ZENO_WITH_TINY_CUDA_NN requires the tiny-cuda-nn target to be provided by the parent build")
    endif()
    target_sources(zeno PRIVATE TinyCudaNNRuntime.cu)
    target_link_libraries(zeno PRIVATE tiny-cuda-nn)
    set_target_properties(zeno PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_compile_definitions(zeno PRIVATE ZENO_WITH_TINY_CUDA_NN=1)

    # Use the upstream NanoVDB GPU builders when this OpenVDB/NanoVDB revision
    # provides them. Older vendored revisions keep the neural decoder available
    # but compile a clear runtime-unavailable fallback for the NanoVDB bridge.
    if (ZENO_NANOVDB_INCLUDE_DIR
        AND EXISTS "${ZENO_NANOVDB_INCLUDE_DIR}/nanovdb/tools/cuda/PointsToGrid.cuh"
        AND EXISTS "${ZENO_NANOVDB_INCLUDE_DIR}/nanovdb/tools/cuda/IndexToGrid.cuh"
        AND EXISTS "${ZENO_NANOVDB_INCLUDE_DIR}/nanovdb/cuda/GridHandle.cuh"
        AND EXISTS "${ZENO_NANOVDB_INCLUDE_DIR}/nanovdb/tools/cuda/GridChecksum.cuh")
        target_compile_definitions(zeno PRIVATE ZENO_WITH_NANOVDB_CUDA_BUILDERS=1)
        message(STATUS "NanoVDB CUDA PointsToGrid + IndexToGrid bridge enabled")
    else()
        message(STATUS "NanoVDB CUDA builders unavailable in this header set; device grid bridge will report unavailable")
    endif()
endif()

target_include_directories(zeno PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/Zeno/zeno>)

if (ZENO_INSTALL_TARGET AND ZENO_BUILD_SHARED)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION include/Zeno/zeno)
endif()

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zeno PRIVATE OpenMP::OpenMP_CXX)
endif()

#find_package(Blosc)
#if (TARGET Blosc::blosc)
    #message(STATUS "Found Blosc::blosc")
    #target_link_libraries(zeno PRIVATE Blosc::blosc)
#endif()
find_package(TBB REQUIRED COMPONENTS tbb tbbmalloc)
message(STATUS "Found TBB ${TBB_VERSION}")
target_link_libraries(zeno PRIVATE TBB::tbbmalloc TBB::tbb)

if (ZENO_SYSTEM_OPENVDB)
    find_package(OpenVDB REQUIRED)
    message(STATUS "Found OpenVDB ${OpenVDB_VERSION}")
    target_link_libraries(zeno PRIVATE OpenVDB::openvdb)
else()
    if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/openvdb/CMakeLists.txt)
        message(FATAL_ERROR "openvdb submodule not found! Please run: git submodule update --init --recursive")
    endif()
    set(OPENVDB_BUILD_BINARIES OFF CACHE BOOL "GUN TMD VDB_PRINT")
    set(OPENVDB_CORE_SHARED OFF CACHE BOOL "GUN TMD SHARED")
    set(OPENVDB_CORE_STATIC ON CACHE BOOL "MAKE STATIC GREAT AGAIN")
    set(DISABLE_DEPENDENCY_VERSION_CHECKS ON CACHE BOOL "GUN TMD VERSION")
    #set(USE_EXR ${ZENO_ENABLE_OPENEXR} CACHE BOOL "FUCK FOLLOW MY EXR" FORCE)
    #set(USE_IMATH_HALF ${ZENO_ENABLE_OPENEXR} CACHE BOOL "FUCK FOLLOW MY ILM" FORCE)
    set(USE_EXPLICIT_INSTANTIATION OFF CACHE BOOL "GUN TMD MFC" FORCE)
    add_subdirectory(openvdb)
    target_link_libraries(zeno INTERFACE $<BUILD_INTERFACE:openvdb> PRIVATE openvdb)
endif()
if (WIN32)
  target_compile_options(zeno PRIVATE -DNOMINMAX -D_USE_MATH_DEFINES)
endif()
