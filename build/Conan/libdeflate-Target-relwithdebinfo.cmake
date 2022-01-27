########### VARIABLES #######################################################################
#############################################################################################

set(libdeflate_COMPILE_OPTIONS_RELWITHDEBINFO
    "$<$<COMPILE_LANGUAGE:CXX>:${libdeflate_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>"
    "$<$<COMPILE_LANGUAGE:C>:${libdeflate_COMPILE_OPTIONS_C_RELWITHDEBINFO}>")

set(libdeflate_LINKER_FLAGS_RELWITHDEBINFO
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${libdeflate_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${libdeflate_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${libdeflate_EXE_LINK_FLAGS_RELWITHDEBINFO}>")

set(libdeflate_FRAMEWORKS_FOUND_RELWITHDEBINFO "") # Will be filled later
conan_find_apple_frameworks(libdeflate_FRAMEWORKS_FOUND_RELWITHDEBINFO "${libdeflate_FRAMEWORKS_RELWITHDEBINFO}" "${libdeflate_FRAMEWORK_DIRS_RELWITHDEBINFO}")

# Gather all the libraries that should be linked to the targets (do not touch existing variables)
set(_libdeflate_DEPENDENCIES_RELWITHDEBINFO "${libdeflate_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${libdeflate_SYSTEM_LIBS_RELWITHDEBINFO} ")

set(libdeflate_LIBRARIES_TARGETS_RELWITHDEBINFO "") # Will be filled later
set(libdeflate_LIBRARIES_RELWITHDEBINFO "") # Will be filled later
conan_package_library_targets("${libdeflate_LIBS_RELWITHDEBINFO}"    # libraries
                              "${libdeflate_LIB_DIRS_RELWITHDEBINFO}" # package_libdir
                              "${_libdeflate_DEPENDENCIES_RELWITHDEBINFO}" # deps
                              libdeflate_LIBRARIES_RELWITHDEBINFO   # out_libraries
                              libdeflate_LIBRARIES_TARGETS_RELWITHDEBINFO  # out_libraries_targets
                              "_RELWITHDEBINFO"  # config_suffix
                              "libdeflate")    # package_name

foreach(_FRAMEWORK ${libdeflate_FRAMEWORKS_FOUND_RELWITHDEBINFO})
    list(APPEND libdeflate_LIBRARIES_TARGETS_RELWITHDEBINFO ${_FRAMEWORK})
    list(APPEND libdeflate_LIBRARIES_RELWITHDEBINFO ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${libdeflate_SYSTEM_LIBS_RELWITHDEBINFO})
    list(APPEND libdeflate_LIBRARIES_TARGETS_RELWITHDEBINFO ${_SYSTEM_LIB})
    list(APPEND libdeflate_LIBRARIES_RELWITHDEBINFO ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(libdeflate_LIBRARIES_TARGETS_RELWITHDEBINFO "${libdeflate_LIBRARIES_TARGETS_RELWITHDEBINFO};")
set(libdeflate_LIBRARIES_RELWITHDEBINFO "${libdeflate_LIBRARIES_RELWITHDEBINFO};")

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${libdeflate_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH ${libdeflate_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_PREFIX_PATH})


########## GLOBAL TARGET PROPERTIES RelWithDebInfo ########################################
set_property(TARGET libdeflate::libdeflate
             PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${libdeflate_LIBRARIES_TARGETS_RELWITHDEBINFO}
                                           ${libdeflate_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET libdeflate::libdeflate
             PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${libdeflate_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET libdeflate::libdeflate
             PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${libdeflate_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET libdeflate::libdeflate
             PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${libdeflate_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET libdeflate::libdeflate
             PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${libdeflate_COMPILE_OPTIONS_RELWITHDEBINFO}> APPEND)

########## COMPONENTS TARGET PROPERTIES RelWithDebInfo ########################################