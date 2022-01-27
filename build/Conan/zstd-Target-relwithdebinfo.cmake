########### VARIABLES #######################################################################
#############################################################################################

set(zstd_COMPILE_OPTIONS_RELWITHDEBINFO
    "$<$<COMPILE_LANGUAGE:CXX>:${zstd_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>"
    "$<$<COMPILE_LANGUAGE:C>:${zstd_COMPILE_OPTIONS_C_RELWITHDEBINFO}>")

set(zstd_LINKER_FLAGS_RELWITHDEBINFO
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${zstd_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${zstd_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${zstd_EXE_LINK_FLAGS_RELWITHDEBINFO}>")

set(zstd_FRAMEWORKS_FOUND_RELWITHDEBINFO "") # Will be filled later
conan_find_apple_frameworks(zstd_FRAMEWORKS_FOUND_RELWITHDEBINFO "${zstd_FRAMEWORKS_RELWITHDEBINFO}" "${zstd_FRAMEWORK_DIRS_RELWITHDEBINFO}")

# Gather all the libraries that should be linked to the targets (do not touch existing variables)
set(_zstd_DEPENDENCIES_RELWITHDEBINFO "${zstd_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${zstd_SYSTEM_LIBS_RELWITHDEBINFO} ")

set(zstd_LIBRARIES_TARGETS_RELWITHDEBINFO "") # Will be filled later
set(zstd_LIBRARIES_RELWITHDEBINFO "") # Will be filled later
conan_package_library_targets("${zstd_LIBS_RELWITHDEBINFO}"    # libraries
                              "${zstd_LIB_DIRS_RELWITHDEBINFO}" # package_libdir
                              "${_zstd_DEPENDENCIES_RELWITHDEBINFO}" # deps
                              zstd_LIBRARIES_RELWITHDEBINFO   # out_libraries
                              zstd_LIBRARIES_TARGETS_RELWITHDEBINFO  # out_libraries_targets
                              "_RELWITHDEBINFO"  # config_suffix
                              "zstd")    # package_name

foreach(_FRAMEWORK ${zstd_FRAMEWORKS_FOUND_RELWITHDEBINFO})
    list(APPEND zstd_LIBRARIES_TARGETS_RELWITHDEBINFO ${_FRAMEWORK})
    list(APPEND zstd_LIBRARIES_RELWITHDEBINFO ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${zstd_SYSTEM_LIBS_RELWITHDEBINFO})
    list(APPEND zstd_LIBRARIES_TARGETS_RELWITHDEBINFO ${_SYSTEM_LIB})
    list(APPEND zstd_LIBRARIES_RELWITHDEBINFO ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(zstd_LIBRARIES_TARGETS_RELWITHDEBINFO "${zstd_LIBRARIES_TARGETS_RELWITHDEBINFO};")
set(zstd_LIBRARIES_RELWITHDEBINFO "${zstd_LIBRARIES_RELWITHDEBINFO};")

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${zstd_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH ${zstd_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_PREFIX_PATH})
########## COMPONENT zstd::libzstd_static FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(zstd_zstd_libzstd_static_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(zstd_zstd_libzstd_static_FRAMEWORKS_FOUND_RELWITHDEBINFO "${zstd_zstd_libzstd_static_FRAMEWORKS_RELWITHDEBINFO}" "${zstd_zstd_libzstd_static_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(zstd_zstd_libzstd_static_LIB_TARGETS_RELWITHDEBINFO "")
set(zstd_zstd_libzstd_static_NOT_USED_RELWITHDEBINFO "")
set(zstd_zstd_libzstd_static_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${zstd_zstd_libzstd_static_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${zstd_zstd_libzstd_static_SYSTEM_LIBS_RELWITHDEBINFO} ${zstd_zstd_libzstd_static_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${zstd_zstd_libzstd_static_LIBS_RELWITHDEBINFO}"
                              "${zstd_zstd_libzstd_static_LIB_DIRS_RELWITHDEBINFO}"
                              "${zstd_zstd_libzstd_static_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              zstd_zstd_libzstd_static_NOT_USED_RELWITHDEBINFO
                              zstd_zstd_libzstd_static_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "zstd_zstd_libzstd_static")

set(zstd_zstd_libzstd_static_LINK_LIBS_RELWITHDEBINFO ${zstd_zstd_libzstd_static_LIB_TARGETS_RELWITHDEBINFO} ${zstd_zstd_libzstd_static_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})


########## GLOBAL TARGET PROPERTIES RelWithDebInfo ########################################
set_property(TARGET zstd::libzstd_static
             PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${zstd_LIBRARIES_TARGETS_RELWITHDEBINFO}
                                           ${zstd_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET zstd::libzstd_static
             PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${zstd_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET zstd::libzstd_static
             PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${zstd_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET zstd::libzstd_static
             PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${zstd_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET zstd::libzstd_static
             PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${zstd_COMPILE_OPTIONS_RELWITHDEBINFO}> APPEND)

########## COMPONENTS TARGET PROPERTIES RelWithDebInfo ########################################
########## COMPONENT zstd::libzstd_static TARGET PROPERTIES ######################################
set_property(TARGET zstd::libzstd_static PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${zstd_zstd_libzstd_static_LINK_LIBS_RELWITHDEBINFO}
             ${zstd_zstd_libzstd_static_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET zstd::libzstd_static PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${zstd_zstd_libzstd_static_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET zstd::libzstd_static PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${zstd_zstd_libzstd_static_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET zstd::libzstd_static PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${zstd_zstd_libzstd_static_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET zstd::libzstd_static PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${zstd_zstd_libzstd_static_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${zstd_zstd_libzstd_static_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(zstd_zstd_libzstd_static_TARGET_PROPERTIES TRUE)