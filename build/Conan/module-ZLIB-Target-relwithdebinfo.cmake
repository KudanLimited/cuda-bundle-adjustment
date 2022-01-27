########### VARIABLES #######################################################################
#############################################################################################

set(zlib_COMPILE_OPTIONS_RELWITHDEBINFO
    "$<$<COMPILE_LANGUAGE:CXX>:${zlib_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>"
    "$<$<COMPILE_LANGUAGE:C>:${zlib_COMPILE_OPTIONS_C_RELWITHDEBINFO}>")

set(zlib_LINKER_FLAGS_RELWITHDEBINFO
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${zlib_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${zlib_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${zlib_EXE_LINK_FLAGS_RELWITHDEBINFO}>")

set(zlib_FRAMEWORKS_FOUND_RELWITHDEBINFO "") # Will be filled later
conan_find_apple_frameworks(zlib_FRAMEWORKS_FOUND_RELWITHDEBINFO "${zlib_FRAMEWORKS_RELWITHDEBINFO}" "${zlib_FRAMEWORK_DIRS_RELWITHDEBINFO}")

# Gather all the libraries that should be linked to the targets (do not touch existing variables)
set(_zlib_DEPENDENCIES_RELWITHDEBINFO "${zlib_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${zlib_SYSTEM_LIBS_RELWITHDEBINFO} ")

set(zlib_LIBRARIES_TARGETS_RELWITHDEBINFO "") # Will be filled later
set(zlib_LIBRARIES_RELWITHDEBINFO "") # Will be filled later
conan_package_library_targets("${zlib_LIBS_RELWITHDEBINFO}"    # libraries
                              "${zlib_LIB_DIRS_RELWITHDEBINFO}" # package_libdir
                              "${_zlib_DEPENDENCIES_RELWITHDEBINFO}" # deps
                              zlib_LIBRARIES_RELWITHDEBINFO   # out_libraries
                              zlib_LIBRARIES_TARGETS_RELWITHDEBINFO  # out_libraries_targets
                              "_RELWITHDEBINFO"  # config_suffix
                              "zlib")    # package_name

foreach(_FRAMEWORK ${zlib_FRAMEWORKS_FOUND_RELWITHDEBINFO})
    list(APPEND zlib_LIBRARIES_TARGETS_RELWITHDEBINFO ${_FRAMEWORK})
    list(APPEND zlib_LIBRARIES_RELWITHDEBINFO ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${zlib_SYSTEM_LIBS_RELWITHDEBINFO})
    list(APPEND zlib_LIBRARIES_TARGETS_RELWITHDEBINFO ${_SYSTEM_LIB})
    list(APPEND zlib_LIBRARIES_RELWITHDEBINFO ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(zlib_LIBRARIES_TARGETS_RELWITHDEBINFO "${zlib_LIBRARIES_TARGETS_RELWITHDEBINFO};")
set(zlib_LIBRARIES_RELWITHDEBINFO "${zlib_LIBRARIES_RELWITHDEBINFO};")

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${zlib_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH ${zlib_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_PREFIX_PATH})


########## GLOBAL TARGET PROPERTIES RelWithDebInfo ########################################
set_property(TARGET ZLIB::ZLIB
             PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${zlib_LIBRARIES_TARGETS_RELWITHDEBINFO}
                                           ${zlib_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET ZLIB::ZLIB
             PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${zlib_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET ZLIB::ZLIB
             PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${zlib_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET ZLIB::ZLIB
             PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${zlib_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET ZLIB::ZLIB
             PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${zlib_COMPILE_OPTIONS_RELWITHDEBINFO}> APPEND)

########## COMPONENTS TARGET PROPERTIES RelWithDebInfo ########################################