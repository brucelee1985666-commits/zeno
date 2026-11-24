# Minimal FindBlosc module for distro OpenVDB packages that ship the
# development headers/library but no CMake package configuration.
find_path(Blosc_INCLUDE_DIR NAMES blosc.h)
find_library(Blosc_LIBRARY NAMES blosc)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Blosc
    REQUIRED_VARS Blosc_INCLUDE_DIR Blosc_LIBRARY)

if(Blosc_FOUND AND NOT TARGET Blosc::blosc)
    add_library(Blosc::blosc UNKNOWN IMPORTED)
    set_target_properties(Blosc::blosc PROPERTIES
        IMPORTED_LOCATION "${Blosc_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${Blosc_INCLUDE_DIR}")
endif()
