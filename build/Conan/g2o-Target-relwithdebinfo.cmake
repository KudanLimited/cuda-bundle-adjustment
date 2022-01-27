########### VARIABLES #######################################################################
#############################################################################################

set(g2o_COMPILE_OPTIONS_RELWITHDEBINFO
    "$<$<COMPILE_LANGUAGE:CXX>:${g2o_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>"
    "$<$<COMPILE_LANGUAGE:C>:${g2o_COMPILE_OPTIONS_C_RELWITHDEBINFO}>")

set(g2o_LINKER_FLAGS_RELWITHDEBINFO
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${g2o_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${g2o_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${g2o_EXE_LINK_FLAGS_RELWITHDEBINFO}>")

set(g2o_FRAMEWORKS_FOUND_RELWITHDEBINFO "") # Will be filled later
conan_find_apple_frameworks(g2o_FRAMEWORKS_FOUND_RELWITHDEBINFO "${g2o_FRAMEWORKS_RELWITHDEBINFO}" "${g2o_FRAMEWORK_DIRS_RELWITHDEBINFO}")

# Gather all the libraries that should be linked to the targets (do not touch existing variables)
set(_g2o_DEPENDENCIES_RELWITHDEBINFO "${g2o_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${g2o_SYSTEM_LIBS_RELWITHDEBINFO} Eigen3::Eigen3")

set(g2o_LIBRARIES_TARGETS_RELWITHDEBINFO "") # Will be filled later
set(g2o_LIBRARIES_RELWITHDEBINFO "") # Will be filled later
conan_package_library_targets("${g2o_LIBS_RELWITHDEBINFO}"    # libraries
                              "${g2o_LIB_DIRS_RELWITHDEBINFO}" # package_libdir
                              "${_g2o_DEPENDENCIES_RELWITHDEBINFO}" # deps
                              g2o_LIBRARIES_RELWITHDEBINFO   # out_libraries
                              g2o_LIBRARIES_TARGETS_RELWITHDEBINFO  # out_libraries_targets
                              "_RELWITHDEBINFO"  # config_suffix
                              "g2o")    # package_name

foreach(_FRAMEWORK ${g2o_FRAMEWORKS_FOUND_RELWITHDEBINFO})
    list(APPEND g2o_LIBRARIES_TARGETS_RELWITHDEBINFO ${_FRAMEWORK})
    list(APPEND g2o_LIBRARIES_RELWITHDEBINFO ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${g2o_SYSTEM_LIBS_RELWITHDEBINFO})
    list(APPEND g2o_LIBRARIES_TARGETS_RELWITHDEBINFO ${_SYSTEM_LIB})
    list(APPEND g2o_LIBRARIES_RELWITHDEBINFO ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(g2o_LIBRARIES_TARGETS_RELWITHDEBINFO "${g2o_LIBRARIES_TARGETS_RELWITHDEBINFO};Eigen3::Eigen3")
set(g2o_LIBRARIES_RELWITHDEBINFO "${g2o_LIBRARIES_RELWITHDEBINFO};Eigen3::Eigen3")

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${g2o_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH ${g2o_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_PREFIX_PATH})


########## GLOBAL TARGET PROPERTIES RelWithDebInfo ########################################
set_property(TARGET G2O::G2O
             PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${g2o_LIBRARIES_TARGETS_RELWITHDEBINFO}
                                           ${g2o_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET G2O::G2O
             PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${g2o_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET G2O::G2O
             PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${g2o_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET G2O::G2O
             PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${g2o_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET G2O::G2O
             PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${g2o_COMPILE_OPTIONS_RELWITHDEBINFO}> APPEND)

########## COMPONENTS TARGET PROPERTIES RelWithDebInfo ########################################