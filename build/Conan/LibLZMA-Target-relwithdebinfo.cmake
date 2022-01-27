########### VARIABLES #######################################################################
#############################################################################################

set(xz_utils_COMPILE_OPTIONS_RELWITHDEBINFO
    "$<$<COMPILE_LANGUAGE:CXX>:${xz_utils_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>"
    "$<$<COMPILE_LANGUAGE:C>:${xz_utils_COMPILE_OPTIONS_C_RELWITHDEBINFO}>")

set(xz_utils_LINKER_FLAGS_RELWITHDEBINFO
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${xz_utils_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${xz_utils_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${xz_utils_EXE_LINK_FLAGS_RELWITHDEBINFO}>")

set(xz_utils_FRAMEWORKS_FOUND_RELWITHDEBINFO "") # Will be filled later
conan_find_apple_frameworks(xz_utils_FRAMEWORKS_FOUND_RELWITHDEBINFO "${xz_utils_FRAMEWORKS_RELWITHDEBINFO}" "${xz_utils_FRAMEWORK_DIRS_RELWITHDEBINFO}")

# Gather all the libraries that should be linked to the targets (do not touch existing variables)
set(_xz_utils_DEPENDENCIES_RELWITHDEBINFO "${xz_utils_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${xz_utils_SYSTEM_LIBS_RELWITHDEBINFO} ")

set(xz_utils_LIBRARIES_TARGETS_RELWITHDEBINFO "") # Will be filled later
set(xz_utils_LIBRARIES_RELWITHDEBINFO "") # Will be filled later
conan_package_library_targets("${xz_utils_LIBS_RELWITHDEBINFO}"    # libraries
                              "${xz_utils_LIB_DIRS_RELWITHDEBINFO}" # package_libdir
                              "${_xz_utils_DEPENDENCIES_RELWITHDEBINFO}" # deps
                              xz_utils_LIBRARIES_RELWITHDEBINFO   # out_libraries
                              xz_utils_LIBRARIES_TARGETS_RELWITHDEBINFO  # out_libraries_targets
                              "_RELWITHDEBINFO"  # config_suffix
                              "xz_utils")    # package_name

foreach(_FRAMEWORK ${xz_utils_FRAMEWORKS_FOUND_RELWITHDEBINFO})
    list(APPEND xz_utils_LIBRARIES_TARGETS_RELWITHDEBINFO ${_FRAMEWORK})
    list(APPEND xz_utils_LIBRARIES_RELWITHDEBINFO ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${xz_utils_SYSTEM_LIBS_RELWITHDEBINFO})
    list(APPEND xz_utils_LIBRARIES_TARGETS_RELWITHDEBINFO ${_SYSTEM_LIB})
    list(APPEND xz_utils_LIBRARIES_RELWITHDEBINFO ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(xz_utils_LIBRARIES_TARGETS_RELWITHDEBINFO "${xz_utils_LIBRARIES_TARGETS_RELWITHDEBINFO};")
set(xz_utils_LIBRARIES_RELWITHDEBINFO "${xz_utils_LIBRARIES_RELWITHDEBINFO};")

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${xz_utils_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH ${xz_utils_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_PREFIX_PATH})


########## GLOBAL TARGET PROPERTIES RelWithDebInfo ########################################
set_property(TARGET LibLZMA::LibLZMA
             PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${xz_utils_LIBRARIES_TARGETS_RELWITHDEBINFO}
                                           ${xz_utils_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET LibLZMA::LibLZMA
             PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${xz_utils_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET LibLZMA::LibLZMA
             PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${xz_utils_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET LibLZMA::LibLZMA
             PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${xz_utils_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET LibLZMA::LibLZMA
             PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${xz_utils_COMPILE_OPTIONS_RELWITHDEBINFO}> APPEND)

########## COMPONENTS TARGET PROPERTIES RelWithDebInfo ########################################