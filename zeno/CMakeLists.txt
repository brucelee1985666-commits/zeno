option(ZENO_BENCHMARKING "Enable ZENO benchmarking timer" ON)
option(ZENO_PARALLEL_STL "Enable parallel STL in ZENO" OFF)
option(ZENO_ENABLE_OPENMP "Enable OpenMP in ZENO for parallelism" ON)
option(ZENO_ENABLE_MAGICENUM "Enable magicenum in ZENO for enum reflection" OFF)
option(ZENO_ENABLE_BACKWARD "Enable ZENO fault handler for traceback" OFF)
option(ZENO_AETHER_ENABLE_PROFILER "Enable thread-safe per-node execution profiling" OFF)
option(ZENO_AETHER_ENABLE_SMART_SCHEDULER "Enable conservative history-based graph root scheduling" OFF)
option(ZENO_AETHER_ENABLE_TASKFLOW "Enable the optional Taskflow 3.11 CPU task-graph backend" OFF)
option(ZENO_AETHER_BUILD_TASKFLOW_TESTS "Build the optional Aether Taskflow smoke test" OFF)

file(GLOB_RECURSE source CONFIGURE_DEPENDS include/*.h src/*.cpp)

if (ZENO_BUILD_SHARED)
    add_library(zeno SHARED ${source})
else()
    add_library(zeno OBJECT ${source})
endif()

if (ZENO_ENABLE_OPENMP)
    find_package(OpenMP)
    if (TARGET OpenMP::OpenMP_CXX)
        message(STATUS "Found OpenMP::OpenMP_CXX")
        target_link_libraries(zeno PRIVATE OpenMP::OpenMP_CXX)
        if (MSVC)
            include(CheckCXXCompilerFlag)
            check_cxx_compiler_flag("/openmp:experimental" MSVC_SUPPORT_OPENMP_EXPERIMENTAL)
            if (MSVC_SUPPORT_OPENMP_EXPERIMENTAL)
                target_compile_options(zeno PRIVATE $<IF:$<COMPILE_LANGUAGE:CXX>,/openmp:experimental,>)
            endif()
        endif()
    else()
        message(WARNING "Not found OpenMP")
    endif()
endif()


if (ZENO_AETHER_ENABLE_TASKFLOW)
    find_path(ZENO_AETHER_TASKFLOW_INCLUDE_DIR
        NAMES taskflow/taskflow.hpp
        HINTS ENV TASKFLOW_ROOT ${TASKFLOW_ROOT}
        PATH_SUFFIXES include
    )
    if (NOT ZENO_AETHER_TASKFLOW_INCLUDE_DIR)
        message(FATAL_ERROR
            "ZENO_AETHER_ENABLE_TASKFLOW=ON, but taskflow/taskflow.hpp was not found. "
            "Install Taskflow v3.11.x and set TASKFLOW_ROOT or ZENO_AETHER_TASKFLOW_INCLUDE_DIR.")
    endif()
    find_package(Threads REQUIRED)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES ${ZENO_AETHER_TASKFLOW_INCLUDE_DIR})
    set(CMAKE_REQUIRED_LIBRARIES Threads::Threads)
    check_cxx_source_compiles("
        #include <taskflow/taskflow.hpp>
        #if TF_VERSION < 301100 || TF_VERSION >= 400000
        #error Unsupported Taskflow version: Aether currently targets 3.11.x (C++17)
        #endif
        int main() { tf::Executor executor; tf::Taskflow taskflow; executor.run(taskflow).get(); }
    " ZENO_AETHER_TASKFLOW_COMPATIBLE)
    unset(CMAKE_REQUIRED_INCLUDES)
    unset(CMAKE_REQUIRED_LIBRARIES)
    if (NOT ZENO_AETHER_TASKFLOW_COMPATIBLE)
        message(FATAL_ERROR "Aether requires Taskflow 3.11.x and a working C++17 toolchain.")
    endif()
    target_include_directories(zeno SYSTEM PRIVATE ${ZENO_AETHER_TASKFLOW_INCLUDE_DIR})
    target_link_libraries(zeno PRIVATE Threads::Threads)
    target_compile_definitions(zeno PUBLIC ZENO_AETHER_ENABLE_TASKFLOW=1)
    message(STATUS "Zeno Aether: Taskflow CPU task-graph backend enabled")
endif()

if (ZENO_AETHER_ENABLE_TASKFLOW AND ZENO_AETHER_BUILD_TASKFLOW_TESTS)
    enable_testing()
    add_executable(zeno_aether_taskflow_smoke tests/AetherTaskflowExecutorSmoke.cpp)
    target_link_libraries(zeno_aether_taskflow_smoke PRIVATE zeno)
    add_test(NAME zeno_aether_taskflow_smoke COMMAND zeno_aether_taskflow_smoke)
endif()

if (ZENO_AETHER_ENABLE_PROFILER OR ZENO_AETHER_ENABLE_SMART_SCHEDULER)
    target_compile_definitions(zeno PUBLIC ZENO_AETHER_ENABLE_PROFILER=1)
endif()

if (ZENO_AETHER_ENABLE_SMART_SCHEDULER)
    target_compile_definitions(zeno PUBLIC ZENO_AETHER_ENABLE_SMART_SCHEDULER=1)
endif()

if (ZENO_BENCHMARKING)
    target_compile_definitions(zeno PUBLIC -DZENO_BENCHMARKING)
endif()

# only work without CUDA option.
#if (ZENO_DEBUG_MSVC)
#    zeno_dbg_msvc(zeno)
#endif()

if (ZENO_WITH_PYTHON3)
    message(STATUS "python3 libraries for zenoedit: ${Python3_LIBRARIES}")
    message(STATUS "python3 include directories for zenoedit: ${Python3_INCLUDE_DIRS}")
    target_link_libraries(zeno PRIVATE ${Python3_LIBRARIES})
    target_include_directories(zeno PRIVATE ${Python3_INCLUDE_DIRS})
endif()

if (ZENO_PARALLEL_STL)
    find_package(Threads REQUIRED)
    target_link_libraries(zeno PRIVATE Threads::Threads)
    if (NOT MSVC)
        find_package(TBB)
        if (TBB_FOUND)
            message(STATUS "Found TBB: ${TBB_DIR}")
            target_link_libraries(zeno PRIVATE TBB::tbb)
        else()
            message(WARNING "TBB not found, disabling parallel STL")
            set(ZENO_PARALLEL_STL OFF)
        endif()
    endif()
    if (ZENO_PARALLEL_STL)
        target_compile_definitions(zeno PUBLIC -DZENO_PARALLEL_STL)
    endif()
endif()

if (ZENO_ENABLE_BACKWARD)
    add_subdirectory(tpls/backward-cpp)
    target_compile_definitions(zeno PUBLIC -DZENO_ENABLE_BACKWARD)
    target_sources(zeno PRIVATE ${BACKWARD_ENABLE})
    add_backward(zeno)
endif()

if (ZENO_BUILD_SHARED)
    target_compile_definitions(zeno PRIVATE -DZENO_DLLEXPORT INTERFACE -DZENO_DLLIMPORT)
endif()

target_include_directories(zeno PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tpls/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tpls/deprecated>
    $<INSTALL_INTERFACE:include/Zeno/zeno/include>)

if (ZENO_INSTALL_TARGET AND ZENO_BUILD_SHARED)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION include/Zeno/zeno)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tpls/include DESTINATION include/Zeno/zeno)
    install(TARGETS zeno EXPORT ZenoTargets)
endif()

if (ZENO_ENABLE_MAGICENUM)
    target_compile_definitions(zeno PUBLIC -DZENO_ENABLE_MAGICENUM)
endif()

#if (ZENO_NO_WARNING)
    #if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        #target_compile_options(zeno PUBLIC $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:-Wno-all -Wno-cpp -Wno-deprecated-declarations -Wno-enum-compare -Wno-ignored-attributes -Wno-extra -Wreturn-type -Wmissing-declarations -Wnon-virtual-dtor -Wsuggest-override -Wconversion-null>>)
    #endif()
    #if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        #target_compile_options(zeno PUBLIC $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:C,CXX>:-w>>)
    #endif()
#endif()
