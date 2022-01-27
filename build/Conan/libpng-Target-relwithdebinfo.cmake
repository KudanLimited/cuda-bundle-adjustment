########### VARIABLES #######################################################################
#############################################################################################

set(libpng_COMPILE_OPTIONS_RELWITHDEBINFO
    "$<$<COMPILE_LANGUAGE:CXX>:${libpng_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>"
    "$<$<COMPILE_LANGUAGE:C>:${libpng_COMPILE_OPTIONS_C_RELWITHDEBINFO}>")

set(libpng_LINKER_FLAGS_RELWITHDEBINFO
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${libpng_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${libpng_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${libpng_EXE_LINK_FLAGS_RELWITHDEBINFO}>")

set(libpng_FRAMEWORKS_FOUND_RELWITHDEBINFO "") # Will be filled later
conan_find_apple_frameworks(libpng_FRAMEWORKS_FOUND_RELWITHDEBINFO "${libpng_FRAMEWORKS_RELWITHDEBINFO}" "${libpng_FRAMEWORK_DIRS_RELWITHDEBINFO}")

# Gather all the libraries that should be linked to the targets (do not touch existing variables)
set(_libpng_DEPENDENCIES_RELWITHDEBINFO "${libpng_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${libpng_SYSTEM_LIBS_RELWITHDEBINFO} ZLIB::ZLIB")

set(libpng_LIBRARIES_TARGETS_RELWITHDEBINFO "") # Will be filled later
set(libpng_LIBRARIES_RELWITHDEBINFO "") # Will be filled later
conan_package_library_targets("${libpng_LIBS_RELWITHDEBINFO}"    # libraries
                              "${libpng_LIB_DIRS_RELWITHDEBINFO}" # package_libdir
                              "${_libpng_DEPENDENCIES_RELWITHDEBINFO}" # deps
                              libpng_LIBRARIES_RELWITHDEBINFO   # out_libraries
                              libpng_LIBRARIES_TARGETS_RELWITHDEBINFO  # out_libraries_targets
                              "_RELWITHDEBINFO"  # config_suffix
                              "libpng")    # package_name

foreach(_FRAMEWORK ${libpng_FRAMEWORKS_FOUND_RELWITHDEBINFO})
    list(APPEND libpng_LIBRARIES_TARGETS_RELWITHDEBINFO ${_FRAMEWORK})
    list(APPEND libpng_LIBRARIES_RELWITHDEBINFO ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${libpng_SYSTEM_LIBS_RELWITHDEBINFO})
    list(APPEND libpng_LIBRARIES_TARGETS_RELWITHDEBINFO ${_SYSTEM_LIB})
    list(APPEND libpng_LIBRARIES_RELWITHDEBINFO ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(libpng_LIBRARIES_TARGETS_RELWITHDEBINFO "${libpng_LIBRARIES_TARGETS_RELWITHDEBINFO};ZLIB::ZLIB")
set(libpng_LIBRARIES_RELWITHDEBINFO "${libpng_LIBRARIES_RELWITHDEBINFO};ZLIB::ZLIB")

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${libpng_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH ${libpng_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_PREFIX_PATH})


########## GLOBAL TARGET PROPERTIES RelWithDebInfo ########################################
set_property(TARGET PNG::PNG
             PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${libpng_LIBRARIES_TARGETS_RELWITHDEBINFO}
                                           ${libpng_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET PNG::PNG
             PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${libpng_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET PNG::PNG
             PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${libpng_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET PNG::PNG
             PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${libpng_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET PNG::PNG
             PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${libpng_COMPILE_OPTIONS_RELWITHDEBINFO}> APPEND)

########## COMPONENTS TARGET PROPERTIES RelWithDebInfo ########################################