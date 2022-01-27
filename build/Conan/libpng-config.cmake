########## MACROS ###########################################################################
#############################################################################################

# Requires CMake > 3.15
if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeDeps' generator only works with CMake >= 3.15")
endif()


include(${CMAKE_CURRENT_LIST_DIR}/cmakedeps_macros.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/libpngTargets.cmake)
include(CMakeFindDependencyMacro)

foreach(_DEPENDENCY ${libpng_FIND_DEPENDENCY_NAMES} )
    if(NOT ${_DEPENDENCY}_FOUND)
        find_dependency(${_DEPENDENCY} REQUIRED NO_MODULE)
    endif()
endforeach()

# Only the first installed configuration is included to avoid the collision
foreach(_BUILD_MODULE ${libpng_BUILD_MODULES_PATHS_RELWITHDEBINFO} )
    conan_message(STATUS "Conan: Including build module from '${_BUILD_MODULE}'")
    include(${_BUILD_MODULE})
endforeach()

