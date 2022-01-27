########## MACROS ###########################################################################
#############################################################################################

# Requires CMake > 3.15
if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeDeps' generator only works with CMake >= 3.15")
endif()

include(FindPackageHandleStandardArgs)
set(jbig_FOUND 1)
set(jbig_VERSION "20160605")

find_package_handle_standard_args(jbig
                                  REQUIRED_VARS jbig_VERSION
                                  VERSION_VAR jbig_VERSION)
mark_as_advanced(jbig_FOUND jbig_VERSION)

include(${CMAKE_CURRENT_LIST_DIR}/cmakedeps_macros.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/module-JBIGTargets.cmake)
include(CMakeFindDependencyMacro)

foreach(_DEPENDENCY ${jbig_FIND_DEPENDENCY_NAMES} )
    if(NOT ${_DEPENDENCY}_FOUND)
        find_dependency(${_DEPENDENCY} REQUIRED NO_MODULE)
    endif()
endforeach()

# Only the first installed configuration is included to avoid the collision
foreach(_BUILD_MODULE ${jbig_BUILD_MODULES_PATHS_RELWITHDEBINFO} )
    conan_message(STATUS "Conan: Including build module from '${_BUILD_MODULE}'")
    include(${_BUILD_MODULE})
endforeach()

