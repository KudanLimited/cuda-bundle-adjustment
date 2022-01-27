########### VARIABLES #######################################################################
#############################################################################################

set(jasper_COMPILE_OPTIONS_RELWITHDEBINFO
    "$<$<COMPILE_LANGUAGE:CXX>:${jasper_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>"
    "$<$<COMPILE_LANGUAGE:C>:${jasper_COMPILE_OPTIONS_C_RELWITHDEBINFO}>")

set(jasper_LINKER_FLAGS_RELWITHDEBINFO
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${jasper_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${jasper_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${jasper_EXE_LINK_FLAGS_RELWITHDEBINFO}>")

set(jasper_FRAMEWORKS_FOUND_RELWITHDEBINFO "") # Will be filled later
conan_find_apple_frameworks(jasper_FRAMEWORKS_FOUND_RELWITHDEBINFO "${jasper_FRAMEWORKS_RELWITHDEBINFO}" "${jasper_FRAMEWORK_DIRS_RELWITHDEBINFO}")

# Gather all the libraries that should be linked to the targets (do not touch existing variables)
set(_jasper_DEPENDENCIES_RELWITHDEBINFO "${jasper_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${jasper_SYSTEM_LIBS_RELWITHDEBINFO} JPEG::JPEG")

set(jasper_LIBRARIES_TARGETS_RELWITHDEBINFO "") # Will be filled later
set(jasper_LIBRARIES_RELWITHDEBINFO "") # Will be filled later
conan_package_library_targets("${jasper_LIBS_RELWITHDEBINFO}"    # libraries
                              "${jasper_LIB_DIRS_RELWITHDEBINFO}" # package_libdir
                              "${_jasper_DEPENDENCIES_RELWITHDEBINFO}" # deps
                              jasper_LIBRARIES_RELWITHDEBINFO   # out_libraries
                              jasper_LIBRARIES_TARGETS_RELWITHDEBINFO  # out_libraries_targets
                              "_RELWITHDEBINFO"  # config_suffix
                              "jasper")    # package_name

foreach(_FRAMEWORK ${jasper_FRAMEWORKS_FOUND_RELWITHDEBINFO})
    list(APPEND jasper_LIBRARIES_TARGETS_RELWITHDEBINFO ${_FRAMEWORK})
    list(APPEND jasper_LIBRARIES_RELWITHDEBINFO ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${jasper_SYSTEM_LIBS_RELWITHDEBINFO})
    list(APPEND jasper_LIBRARIES_TARGETS_RELWITHDEBINFO ${_SYSTEM_LIB})
    list(APPEND jasper_LIBRARIES_RELWITHDEBINFO ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(jasper_LIBRARIES_TARGETS_RELWITHDEBINFO "${jasper_LIBRARIES_TARGETS_RELWITHDEBINFO};JPEG::JPEG")
set(jasper_LIBRARIES_RELWITHDEBINFO "${jasper_LIBRARIES_RELWITHDEBINFO};JPEG::JPEG")

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${jasper_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH ${jasper_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_PREFIX_PATH})


########## GLOBAL TARGET PROPERTIES RelWithDebInfo ########################################
set_property(TARGET Jasper::Jasper
             PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${jasper_LIBRARIES_TARGETS_RELWITHDEBINFO}
                                           ${jasper_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET Jasper::Jasper
             PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${jasper_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET Jasper::Jasper
             PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${jasper_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET Jasper::Jasper
             PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${jasper_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET Jasper::Jasper
             PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${jasper_COMPILE_OPTIONS_RELWITHDEBINFO}> APPEND)

########## COMPONENTS TARGET PROPERTIES RelWithDebInfo ########################################