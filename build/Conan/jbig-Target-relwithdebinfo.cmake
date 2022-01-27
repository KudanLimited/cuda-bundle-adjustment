########### VARIABLES #######################################################################
#############################################################################################

set(jbig_COMPILE_OPTIONS_RELWITHDEBINFO
    "$<$<COMPILE_LANGUAGE:CXX>:${jbig_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>"
    "$<$<COMPILE_LANGUAGE:C>:${jbig_COMPILE_OPTIONS_C_RELWITHDEBINFO}>")

set(jbig_LINKER_FLAGS_RELWITHDEBINFO
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${jbig_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${jbig_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${jbig_EXE_LINK_FLAGS_RELWITHDEBINFO}>")

set(jbig_FRAMEWORKS_FOUND_RELWITHDEBINFO "") # Will be filled later
conan_find_apple_frameworks(jbig_FRAMEWORKS_FOUND_RELWITHDEBINFO "${jbig_FRAMEWORKS_RELWITHDEBINFO}" "${jbig_FRAMEWORK_DIRS_RELWITHDEBINFO}")

# Gather all the libraries that should be linked to the targets (do not touch existing variables)
set(_jbig_DEPENDENCIES_RELWITHDEBINFO "${jbig_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${jbig_SYSTEM_LIBS_RELWITHDEBINFO} ")

set(jbig_LIBRARIES_TARGETS_RELWITHDEBINFO "") # Will be filled later
set(jbig_LIBRARIES_RELWITHDEBINFO "") # Will be filled later
conan_package_library_targets("${jbig_LIBS_RELWITHDEBINFO}"    # libraries
                              "${jbig_LIB_DIRS_RELWITHDEBINFO}" # package_libdir
                              "${_jbig_DEPENDENCIES_RELWITHDEBINFO}" # deps
                              jbig_LIBRARIES_RELWITHDEBINFO   # out_libraries
                              jbig_LIBRARIES_TARGETS_RELWITHDEBINFO  # out_libraries_targets
                              "_RELWITHDEBINFO"  # config_suffix
                              "jbig")    # package_name

foreach(_FRAMEWORK ${jbig_FRAMEWORKS_FOUND_RELWITHDEBINFO})
    list(APPEND jbig_LIBRARIES_TARGETS_RELWITHDEBINFO ${_FRAMEWORK})
    list(APPEND jbig_LIBRARIES_RELWITHDEBINFO ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${jbig_SYSTEM_LIBS_RELWITHDEBINFO})
    list(APPEND jbig_LIBRARIES_TARGETS_RELWITHDEBINFO ${_SYSTEM_LIB})
    list(APPEND jbig_LIBRARIES_RELWITHDEBINFO ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(jbig_LIBRARIES_TARGETS_RELWITHDEBINFO "${jbig_LIBRARIES_TARGETS_RELWITHDEBINFO};")
set(jbig_LIBRARIES_RELWITHDEBINFO "${jbig_LIBRARIES_RELWITHDEBINFO};")

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${jbig_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH ${jbig_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_PREFIX_PATH})


########## GLOBAL TARGET PROPERTIES RelWithDebInfo ########################################
set_property(TARGET JBIG::JBIG
             PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${jbig_LIBRARIES_TARGETS_RELWITHDEBINFO}
                                           ${jbig_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET JBIG::JBIG
             PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${jbig_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET JBIG::JBIG
             PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${jbig_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET JBIG::JBIG
             PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${jbig_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET JBIG::JBIG
             PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${jbig_COMPILE_OPTIONS_RELWITHDEBINFO}> APPEND)

########## COMPONENTS TARGET PROPERTIES RelWithDebInfo ########################################