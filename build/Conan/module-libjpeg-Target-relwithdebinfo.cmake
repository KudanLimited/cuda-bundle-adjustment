########### VARIABLES #######################################################################
#############################################################################################

set(libjpeg_COMPILE_OPTIONS_RELWITHDEBINFO
    "$<$<COMPILE_LANGUAGE:CXX>:${libjpeg_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>"
    "$<$<COMPILE_LANGUAGE:C>:${libjpeg_COMPILE_OPTIONS_C_RELWITHDEBINFO}>")

set(libjpeg_LINKER_FLAGS_RELWITHDEBINFO
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${libjpeg_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${libjpeg_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${libjpeg_EXE_LINK_FLAGS_RELWITHDEBINFO}>")

set(libjpeg_FRAMEWORKS_FOUND_RELWITHDEBINFO "") # Will be filled later
conan_find_apple_frameworks(libjpeg_FRAMEWORKS_FOUND_RELWITHDEBINFO "${libjpeg_FRAMEWORKS_RELWITHDEBINFO}" "${libjpeg_FRAMEWORK_DIRS_RELWITHDEBINFO}")

# Gather all the libraries that should be linked to the targets (do not touch existing variables)
set(_libjpeg_DEPENDENCIES_RELWITHDEBINFO "${libjpeg_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${libjpeg_SYSTEM_LIBS_RELWITHDEBINFO} ")

set(libjpeg_LIBRARIES_TARGETS_RELWITHDEBINFO "") # Will be filled later
set(libjpeg_LIBRARIES_RELWITHDEBINFO "") # Will be filled later
conan_package_library_targets("${libjpeg_LIBS_RELWITHDEBINFO}"    # libraries
                              "${libjpeg_LIB_DIRS_RELWITHDEBINFO}" # package_libdir
                              "${_libjpeg_DEPENDENCIES_RELWITHDEBINFO}" # deps
                              libjpeg_LIBRARIES_RELWITHDEBINFO   # out_libraries
                              libjpeg_LIBRARIES_TARGETS_RELWITHDEBINFO  # out_libraries_targets
                              "_RELWITHDEBINFO"  # config_suffix
                              "libjpeg")    # package_name

foreach(_FRAMEWORK ${libjpeg_FRAMEWORKS_FOUND_RELWITHDEBINFO})
    list(APPEND libjpeg_LIBRARIES_TARGETS_RELWITHDEBINFO ${_FRAMEWORK})
    list(APPEND libjpeg_LIBRARIES_RELWITHDEBINFO ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${libjpeg_SYSTEM_LIBS_RELWITHDEBINFO})
    list(APPEND libjpeg_LIBRARIES_TARGETS_RELWITHDEBINFO ${_SYSTEM_LIB})
    list(APPEND libjpeg_LIBRARIES_RELWITHDEBINFO ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(libjpeg_LIBRARIES_TARGETS_RELWITHDEBINFO "${libjpeg_LIBRARIES_TARGETS_RELWITHDEBINFO};")
set(libjpeg_LIBRARIES_RELWITHDEBINFO "${libjpeg_LIBRARIES_RELWITHDEBINFO};")

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${libjpeg_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH ${libjpeg_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_PREFIX_PATH})


########## GLOBAL TARGET PROPERTIES RelWithDebInfo ########################################
set_property(TARGET JPEG::JPEG
             PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${libjpeg_LIBRARIES_TARGETS_RELWITHDEBINFO}
                                           ${libjpeg_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET JPEG::JPEG
             PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${libjpeg_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET JPEG::JPEG
             PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${libjpeg_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET JPEG::JPEG
             PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${libjpeg_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET JPEG::JPEG
             PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${libjpeg_COMPILE_OPTIONS_RELWITHDEBINFO}> APPEND)

########## COMPONENTS TARGET PROPERTIES RelWithDebInfo ########################################