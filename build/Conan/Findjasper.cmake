########## MACROS ###########################################################################
#############################################################################################

# Requires CMake > 3.15
if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeDeps' generator only works with CMake >= 3.15")
endif()

include(FindPackageHandleStandardArgs)
set(jasper_FOUND 1)
set(jasper_VERSION "2.0.32")

find_package_handle_standard_args(jasper
                                  REQUIRED_VARS jasper_VERSION
                                  VERSION_VAR jasper_VERSION)
mark_as_advanced(jasper_FOUND jasper_VERSION)

include(${CMAKE_CURRENT_LIST_DIR}/cmakedeps_macros.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/module-jasperTargets.cmake)
include(CMakeFindDependencyMacro)

foreach(_DEPENDENCY ${jasper_FIND_DEPENDENCY_NAMES} )
    if(NOT ${_DEPENDENCY}_FOUND)
        find_dependency(${_DEPENDENCY} REQUIRED NO_MODULE)
    endif()
endforeach()

# Only the first installed configuration is included to avoid the collision
foreach(_BUILD_MODULE ${jasper_BUILD_MODULES_PATHS_RELWITHDEBINFO} )
    conan_message(STATUS "Conan: Including build module from '${_BUILD_MODULE}'")
    include(${_BUILD_MODULE})
endforeach()

