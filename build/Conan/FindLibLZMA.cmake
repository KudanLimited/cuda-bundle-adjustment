########## MACROS ###########################################################################
#############################################################################################

# Requires CMake > 3.15
if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeDeps' generator only works with CMake >= 3.15")
endif()

include(FindPackageHandleStandardArgs)
set(xz_utils_FOUND 1)
set(xz_utils_VERSION "5.2.5")

find_package_handle_standard_args(xz_utils
                                  REQUIRED_VARS xz_utils_VERSION
                                  VERSION_VAR xz_utils_VERSION)
mark_as_advanced(xz_utils_FOUND xz_utils_VERSION)

include(${CMAKE_CURRENT_LIST_DIR}/cmakedeps_macros.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/module-LibLZMATargets.cmake)
include(CMakeFindDependencyMacro)

foreach(_DEPENDENCY ${xz_utils_FIND_DEPENDENCY_NAMES} )
    if(NOT ${_DEPENDENCY}_FOUND)
        find_dependency(${_DEPENDENCY} REQUIRED NO_MODULE)
    endif()
endforeach()

# Only the first installed configuration is included to avoid the collision
foreach(_BUILD_MODULE ${xz_utils_BUILD_MODULES_PATHS_RELWITHDEBINFO} )
    conan_message(STATUS "Conan: Including build module from '${_BUILD_MODULE}'")
    include(${_BUILD_MODULE})
endforeach()

