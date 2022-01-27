########## MACROS ###########################################################################
#############################################################################################

# Requires CMake > 3.15
if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeDeps' generator only works with CMake >= 3.15")
endif()

include(FindPackageHandleStandardArgs)
set(eigen_FOUND 1)
set(eigen_VERSION "3.3.9")

find_package_handle_standard_args(eigen
                                  REQUIRED_VARS eigen_VERSION
                                  VERSION_VAR eigen_VERSION)
mark_as_advanced(eigen_FOUND eigen_VERSION)

include(${CMAKE_CURRENT_LIST_DIR}/cmakedeps_macros.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/module-eigenTargets.cmake)
include(CMakeFindDependencyMacro)

foreach(_DEPENDENCY ${eigen_FIND_DEPENDENCY_NAMES} )
    if(NOT ${_DEPENDENCY}_FOUND)
        find_dependency(${_DEPENDENCY} REQUIRED NO_MODULE)
    endif()
endforeach()

# Only the first installed configuration is included to avoid the collision
foreach(_BUILD_MODULE ${eigen_BUILD_MODULES_PATHS_RELWITHDEBINFO} )
    conan_message(STATUS "Conan: Including build module from '${_BUILD_MODULE}'")
    include(${_BUILD_MODULE})
endforeach()

