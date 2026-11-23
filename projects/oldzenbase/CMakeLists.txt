file(GLOB PROJECT_SOURCE include/*/*.h *.cpp mesh/*.cpp pars/*.cpp prim/*.cpp *.cpp)

target_sources(zeno PRIVATE ${PROJECT_SOURCE})
target_compile_definitions(zeno PRIVATE -DDLL_ZENBASE)
target_include_directories(zeno PRIVATE include)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zeno PRIVATE OpenMP::OpenMP_CXX)
endif()
#find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc)
#target_link_libraries(zeno PRIVATE TBB::tbb TBB::tbbmalloc)

if (ZENO_AETHER_ENABLE_TASKFLOW AND ZENO_AETHER_BUILD_TASKFLOW_TESTS)
    add_executable(zeno_aether_oldzenbase_transform_mesh_smoke
        tests/AetherTransformMeshPreparedSmoke.cpp)
    target_include_directories(zeno_aether_oldzenbase_transform_mesh_smoke PRIVATE include)
    target_link_libraries(zeno_aether_oldzenbase_transform_mesh_smoke PRIVATE zeno)
    add_test(NAME zeno_aether_oldzenbase_transform_mesh_smoke
        COMMAND zeno_aether_oldzenbase_transform_mesh_smoke)
endif()
