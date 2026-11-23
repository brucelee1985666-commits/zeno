# Optional integration points for the Zeno Aether roadmap.
# All switches default to OFF so existing Zeno builds remain unchanged.
#
# This file wires dependencies only. It deliberately does not claim that a
# scheduler, Vulkan renderer, or NanoVDB node implementation exists yet.

option(ZENO_AETHER_ENABLE_TBB_RUNTIME
    "Expose oneTBB to the Zeno runtime for the upcoming task-scheduler integration" OFF)
option(ZENO_AETHER_ENABLE_NANOVDB
    "Enable NanoVDB headers for GPU-resident sparse-volume work" OFF)
option(ZENO_AETHER_ENABLE_VULKAN
    "Link Vulkan into the Zeno viewport for the future cross-vendor backend" OFF)

if (ZENO_AETHER_ENABLE_TBB_RUNTIME)
    find_package(TBB 2021 REQUIRED COMPONENTS tbb)
    target_link_libraries(zeno PRIVATE TBB::tbb)
    target_compile_definitions(zeno PRIVATE ZENO_AETHER_HAS_TBB=1)
    message(STATUS "Zeno Aether: oneTBB runtime dependency enabled")
endif()

if (ZENO_AETHER_ENABLE_NANOVDB)
    # NanoVDB is distributed as part of OpenVDB, but its include directory
    # varies by package manager and OpenVDB release. Do not download it here.
    find_path(ZENO_AETHER_NANOVDB_INCLUDE_DIR
        NAMES NanoVDB.h
        HINTS
            ENV OPENVDB_ROOT
            ENV OpenVDB_ROOT
            ${PROJECT_SOURCE_DIR}/projects/zenvdb/openvdb
        PATH_SUFFIXES
            include/nanovdb
            include/openvdb/nanovdb
            nanovdb/nanovdb
            nanovdb
    )
    if (NOT ZENO_AETHER_NANOVDB_INCLUDE_DIR)
        message(FATAL_ERROR
            "ZENO_AETHER_ENABLE_NANOVDB=ON, but NanoVDB.h was not found. "
            "Install NanoVDB from a compatible OpenVDB distribution and set "
            "ZENO_AETHER_NANOVDB_INCLUDE_DIR or OPENVDB_ROOT.")
    endif()
    target_include_directories(zeno PRIVATE ${ZENO_AETHER_NANOVDB_INCLUDE_DIR})
    target_compile_definitions(zeno PRIVATE ZENO_AETHER_HAS_NANOVDB=1)
    message(STATUS "Zeno Aether: NanoVDB headers: ${ZENO_AETHER_NANOVDB_INCLUDE_DIR}")
endif()

if (ZENO_AETHER_ENABLE_VULKAN)
    if (NOT TARGET zenovis)
        message(FATAL_ERROR
            "ZENO_AETHER_ENABLE_VULKAN requires ZENO_BUILD_EDITOR=ON "
            "because the current integration target is zenovis.")
    endif()
    find_package(Vulkan REQUIRED)
    target_link_libraries(zenovis PRIVATE Vulkan::Vulkan)
    target_compile_definitions(zenovis PRIVATE ZENO_AETHER_HAS_VULKAN=1)
    message(STATUS "Zeno Aether: Vulkan SDK linked to zenovis")
endif()
