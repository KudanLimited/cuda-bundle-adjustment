########### VARIABLES #######################################################################
#############################################################################################

set(eigen_COMPILE_OPTIONS_RELWITHDEBINFO
    "$<$<COMPILE_LANGUAGE:CXX>:${eigen_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>"
    "$<$<COMPILE_LANGUAGE:C>:${eigen_COMPILE_OPTIONS_C_RELWITHDEBINFO}>")

set(eigen_LINKER_FLAGS_RELWITHDEBINFO
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${eigen_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${eigen_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${eigen_EXE_LINK_FLAGS_RELWITHDEBINFO}>")

set(eigen_FRAMEWORKS_FOUND_RELWITHDEBINFO "") # Will be filled later
conan_find_apple_frameworks(eigen_FRAMEWORKS_FOUND_RELWITHDEBINFO "${eigen_FRAMEWORKS_RELWITHDEBINFO}" "${eigen_FRAMEWORK_DIRS_RELWITHDEBINFO}")

# Gather all the libraries that should be linked to the targets (do not touch existing variables)
set(_eigen_DEPENDENCIES_RELWITHDEBINFO "${eigen_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${eigen_SYSTEM_LIBS_RELWITHDEBINFO} ")

set(eigen_LIBRARIES_TARGETS_RELWITHDEBINFO "") # Will be filled later
set(eigen_LIBRARIES_RELWITHDEBINFO "") # Will be filled later
conan_package_library_targets("${eigen_LIBS_RELWITHDEBINFO}"    # libraries
                              "${eigen_LIB_DIRS_RELWITHDEBINFO}" # package_libdir
                              "${_eigen_DEPENDENCIES_RELWITHDEBINFO}" # deps
                              eigen_LIBRARIES_RELWITHDEBINFO   # out_libraries
                              eigen_LIBRARIES_TARGETS_RELWITHDEBINFO  # out_libraries_targets
                              "_RELWITHDEBINFO"  # config_suffix
                              "eigen")    # package_name

foreach(_FRAMEWORK ${eigen_FRAMEWORKS_FOUND_RELWITHDEBINFO})
    list(APPEND eigen_LIBRARIES_TARGETS_RELWITHDEBINFO ${_FRAMEWORK})
    list(APPEND eigen_LIBRARIES_RELWITHDEBINFO ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${eigen_SYSTEM_LIBS_RELWITHDEBINFO})
    list(APPEND eigen_LIBRARIES_TARGETS_RELWITHDEBINFO ${_SYSTEM_LIB})
    list(APPEND eigen_LIBRARIES_RELWITHDEBINFO ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(eigen_LIBRARIES_TARGETS_RELWITHDEBINFO "${eigen_LIBRARIES_TARGETS_RELWITHDEBINFO};")
set(eigen_LIBRARIES_RELWITHDEBINFO "${eigen_LIBRARIES_RELWITHDEBINFO};")

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${eigen_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH ${eigen_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_PREFIX_PATH})
########## COMPONENT Eigen3::eigen3 FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(eigen_Eigen3_eigen3_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(eigen_Eigen3_eigen3_FRAMEWORKS_FOUND_RELWITHDEBINFO "${eigen_Eigen3_eigen3_FRAMEWORKS_RELWITHDEBINFO}" "${eigen_Eigen3_eigen3_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(eigen_Eigen3_eigen3_LIB_TARGETS_RELWITHDEBINFO "")
set(eigen_Eigen3_eigen3_NOT_USED_RELWITHDEBINFO "")
set(eigen_Eigen3_eigen3_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${eigen_Eigen3_eigen3_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${eigen_Eigen3_eigen3_SYSTEM_LIBS_RELWITHDEBINFO} ${eigen_Eigen3_eigen3_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${eigen_Eigen3_eigen3_LIBS_RELWITHDEBINFO}"
                              "${eigen_Eigen3_eigen3_LIB_DIRS_RELWITHDEBINFO}"
                              "${eigen_Eigen3_eigen3_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              eigen_Eigen3_eigen3_NOT_USED_RELWITHDEBINFO
                              eigen_Eigen3_eigen3_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "eigen_Eigen3_eigen3")

set(eigen_Eigen3_eigen3_LINK_LIBS_RELWITHDEBINFO ${eigen_Eigen3_eigen3_LIB_TARGETS_RELWITHDEBINFO} ${eigen_Eigen3_eigen3_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})


########## GLOBAL TARGET PROPERTIES RelWithDebInfo ########################################
set_property(TARGET Eigen3::Eigen3
             PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${eigen_LIBRARIES_TARGETS_RELWITHDEBINFO}
                                           ${eigen_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET Eigen3::Eigen3
             PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${eigen_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET Eigen3::Eigen3
             PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${eigen_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET Eigen3::Eigen3
             PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${eigen_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET Eigen3::Eigen3
             PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${eigen_COMPILE_OPTIONS_RELWITHDEBINFO}> APPEND)

########## COMPONENTS TARGET PROPERTIES RelWithDebInfo ########################################
########## COMPONENT Eigen3::eigen3 TARGET PROPERTIES ######################################
set_property(TARGET Eigen3::eigen3 PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${eigen_Eigen3_eigen3_LINK_LIBS_RELWITHDEBINFO}
             ${eigen_Eigen3_eigen3_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET Eigen3::eigen3 PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${eigen_Eigen3_eigen3_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET Eigen3::eigen3 PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${eigen_Eigen3_eigen3_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET Eigen3::eigen3 PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${eigen_Eigen3_eigen3_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET Eigen3::eigen3 PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${eigen_Eigen3_eigen3_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${eigen_Eigen3_eigen3_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(eigen_Eigen3_eigen3_TARGET_PROPERTIES TRUE)