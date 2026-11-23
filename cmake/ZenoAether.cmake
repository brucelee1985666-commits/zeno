# Optional integration points for the Zeno Aether roadmap.
# All switches default to OFF so existing Zeno builds remain unchanged.
#
# This file wires dependencies only. It deliberately does not claim that a
# scheduler, Vulkan renderer, or NanoVDB node implementation exists yet.

option(ZENO_AETHER_ENABLE_NANOVDB
    "Enable NanoVDB headers for GPU-resident sparse-volume work" OFF)


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
