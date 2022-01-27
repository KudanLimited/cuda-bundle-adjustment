########### VARIABLES #######################################################################
#############################################################################################

set(libwebp_COMPILE_OPTIONS_RELWITHDEBINFO
    "$<$<COMPILE_LANGUAGE:CXX>:${libwebp_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>"
    "$<$<COMPILE_LANGUAGE:C>:${libwebp_COMPILE_OPTIONS_C_RELWITHDEBINFO}>")

set(libwebp_LINKER_FLAGS_RELWITHDEBINFO
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${libwebp_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${libwebp_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${libwebp_EXE_LINK_FLAGS_RELWITHDEBINFO}>")

set(libwebp_FRAMEWORKS_FOUND_RELWITHDEBINFO "") # Will be filled later
conan_find_apple_frameworks(libwebp_FRAMEWORKS_FOUND_RELWITHDEBINFO "${libwebp_FRAMEWORKS_RELWITHDEBINFO}" "${libwebp_FRAMEWORK_DIRS_RELWITHDEBINFO}")

# Gather all the libraries that should be linked to the targets (do not touch existing variables)
set(_libwebp_DEPENDENCIES_RELWITHDEBINFO "${libwebp_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${libwebp_SYSTEM_LIBS_RELWITHDEBINFO} libwebp::webp;libwebp::webp;libwebp::webp;libwebp::webp;libwebp::webp;libwebp::webp")

set(libwebp_LIBRARIES_TARGETS_RELWITHDEBINFO "") # Will be filled later
set(libwebp_LIBRARIES_RELWITHDEBINFO "") # Will be filled later
conan_package_library_targets("${libwebp_LIBS_RELWITHDEBINFO}"    # libraries
                              "${libwebp_LIB_DIRS_RELWITHDEBINFO}" # package_libdir
                              "${_libwebp_DEPENDENCIES_RELWITHDEBINFO}" # deps
                              libwebp_LIBRARIES_RELWITHDEBINFO   # out_libraries
                              libwebp_LIBRARIES_TARGETS_RELWITHDEBINFO  # out_libraries_targets
                              "_RELWITHDEBINFO"  # config_suffix
                              "libwebp")    # package_name

foreach(_FRAMEWORK ${libwebp_FRAMEWORKS_FOUND_RELWITHDEBINFO})
    list(APPEND libwebp_LIBRARIES_TARGETS_RELWITHDEBINFO ${_FRAMEWORK})
    list(APPEND libwebp_LIBRARIES_RELWITHDEBINFO ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${libwebp_SYSTEM_LIBS_RELWITHDEBINFO})
    list(APPEND libwebp_LIBRARIES_TARGETS_RELWITHDEBINFO ${_SYSTEM_LIB})
    list(APPEND libwebp_LIBRARIES_RELWITHDEBINFO ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(libwebp_LIBRARIES_TARGETS_RELWITHDEBINFO "${libwebp_LIBRARIES_TARGETS_RELWITHDEBINFO};libwebp::webp;libwebp::webp;libwebp::webp;libwebp::webp;libwebp::webp;libwebp::webp")
set(libwebp_LIBRARIES_RELWITHDEBINFO "${libwebp_LIBRARIES_RELWITHDEBINFO};libwebp::webp;libwebp::webp;libwebp::webp;libwebp::webp;libwebp::webp;libwebp::webp")

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${libwebp_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH ${libwebp_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_PREFIX_PATH})
########## COMPONENT libwebp::webpmux FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(libwebp_libwebp_webpmux_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(libwebp_libwebp_webpmux_FRAMEWORKS_FOUND_RELWITHDEBINFO "${libwebp_libwebp_webpmux_FRAMEWORKS_RELWITHDEBINFO}" "${libwebp_libwebp_webpmux_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(libwebp_libwebp_webpmux_LIB_TARGETS_RELWITHDEBINFO "")
set(libwebp_libwebp_webpmux_NOT_USED_RELWITHDEBINFO "")
set(libwebp_libwebp_webpmux_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${libwebp_libwebp_webpmux_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${libwebp_libwebp_webpmux_SYSTEM_LIBS_RELWITHDEBINFO} ${libwebp_libwebp_webpmux_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${libwebp_libwebp_webpmux_LIBS_RELWITHDEBINFO}"
                              "${libwebp_libwebp_webpmux_LIB_DIRS_RELWITHDEBINFO}"
                              "${libwebp_libwebp_webpmux_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              libwebp_libwebp_webpmux_NOT_USED_RELWITHDEBINFO
                              libwebp_libwebp_webpmux_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "libwebp_libwebp_webpmux")

set(libwebp_libwebp_webpmux_LINK_LIBS_RELWITHDEBINFO ${libwebp_libwebp_webpmux_LIB_TARGETS_RELWITHDEBINFO} ${libwebp_libwebp_webpmux_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT libwebp::webpdemux FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(libwebp_libwebp_webpdemux_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(libwebp_libwebp_webpdemux_FRAMEWORKS_FOUND_RELWITHDEBINFO "${libwebp_libwebp_webpdemux_FRAMEWORKS_RELWITHDEBINFO}" "${libwebp_libwebp_webpdemux_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(libwebp_libwebp_webpdemux_LIB_TARGETS_RELWITHDEBINFO "")
set(libwebp_libwebp_webpdemux_NOT_USED_RELWITHDEBINFO "")
set(libwebp_libwebp_webpdemux_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${libwebp_libwebp_webpdemux_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${libwebp_libwebp_webpdemux_SYSTEM_LIBS_RELWITHDEBINFO} ${libwebp_libwebp_webpdemux_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${libwebp_libwebp_webpdemux_LIBS_RELWITHDEBINFO}"
                              "${libwebp_libwebp_webpdemux_LIB_DIRS_RELWITHDEBINFO}"
                              "${libwebp_libwebp_webpdemux_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              libwebp_libwebp_webpdemux_NOT_USED_RELWITHDEBINFO
                              libwebp_libwebp_webpdemux_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "libwebp_libwebp_webpdemux")

set(libwebp_libwebp_webpdemux_LINK_LIBS_RELWITHDEBINFO ${libwebp_libwebp_webpdemux_LIB_TARGETS_RELWITHDEBINFO} ${libwebp_libwebp_webpdemux_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT libwebp::webp FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(libwebp_libwebp_webp_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(libwebp_libwebp_webp_FRAMEWORKS_FOUND_RELWITHDEBINFO "${libwebp_libwebp_webp_FRAMEWORKS_RELWITHDEBINFO}" "${libwebp_libwebp_webp_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(libwebp_libwebp_webp_LIB_TARGETS_RELWITHDEBINFO "")
set(libwebp_libwebp_webp_NOT_USED_RELWITHDEBINFO "")
set(libwebp_libwebp_webp_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${libwebp_libwebp_webp_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${libwebp_libwebp_webp_SYSTEM_LIBS_RELWITHDEBINFO} ${libwebp_libwebp_webp_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${libwebp_libwebp_webp_LIBS_RELWITHDEBINFO}"
                              "${libwebp_libwebp_webp_LIB_DIRS_RELWITHDEBINFO}"
                              "${libwebp_libwebp_webp_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              libwebp_libwebp_webp_NOT_USED_RELWITHDEBINFO
                              libwebp_libwebp_webp_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "libwebp_libwebp_webp")

set(libwebp_libwebp_webp_LINK_LIBS_RELWITHDEBINFO ${libwebp_libwebp_webp_LIB_TARGETS_RELWITHDEBINFO} ${libwebp_libwebp_webp_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT libwebp::webpdecoder FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(libwebp_libwebp_webpdecoder_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(libwebp_libwebp_webpdecoder_FRAMEWORKS_FOUND_RELWITHDEBINFO "${libwebp_libwebp_webpdecoder_FRAMEWORKS_RELWITHDEBINFO}" "${libwebp_libwebp_webpdecoder_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(libwebp_libwebp_webpdecoder_LIB_TARGETS_RELWITHDEBINFO "")
set(libwebp_libwebp_webpdecoder_NOT_USED_RELWITHDEBINFO "")
set(libwebp_libwebp_webpdecoder_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${libwebp_libwebp_webpdecoder_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${libwebp_libwebp_webpdecoder_SYSTEM_LIBS_RELWITHDEBINFO} ${libwebp_libwebp_webpdecoder_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${libwebp_libwebp_webpdecoder_LIBS_RELWITHDEBINFO}"
                              "${libwebp_libwebp_webpdecoder_LIB_DIRS_RELWITHDEBINFO}"
                              "${libwebp_libwebp_webpdecoder_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              libwebp_libwebp_webpdecoder_NOT_USED_RELWITHDEBINFO
                              libwebp_libwebp_webpdecoder_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "libwebp_libwebp_webpdecoder")

set(libwebp_libwebp_webpdecoder_LINK_LIBS_RELWITHDEBINFO ${libwebp_libwebp_webpdecoder_LIB_TARGETS_RELWITHDEBINFO} ${libwebp_libwebp_webpdecoder_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})


########## GLOBAL TARGET PROPERTIES RelWithDebInfo ########################################
set_property(TARGET libwebp::libwebp
             PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${libwebp_LIBRARIES_TARGETS_RELWITHDEBINFO}
                                           ${libwebp_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET libwebp::libwebp
             PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${libwebp_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET libwebp::libwebp
             PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${libwebp_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET libwebp::libwebp
             PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${libwebp_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET libwebp::libwebp
             PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${libwebp_COMPILE_OPTIONS_RELWITHDEBINFO}> APPEND)

########## COMPONENTS TARGET PROPERTIES RelWithDebInfo ########################################
########## COMPONENT libwebp::webpmux TARGET PROPERTIES ######################################
set_property(TARGET libwebp::webpmux PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${libwebp_libwebp_webpmux_LINK_LIBS_RELWITHDEBINFO}
             ${libwebp_libwebp_webpmux_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET libwebp::webpmux PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${libwebp_libwebp_webpmux_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET libwebp::webpmux PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${libwebp_libwebp_webpmux_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET libwebp::webpmux PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${libwebp_libwebp_webpmux_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET libwebp::webpmux PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${libwebp_libwebp_webpmux_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${libwebp_libwebp_webpmux_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(libwebp_libwebp_webpmux_TARGET_PROPERTIES TRUE)
########## COMPONENT libwebp::webpdemux TARGET PROPERTIES ######################################
set_property(TARGET libwebp::webpdemux PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${libwebp_libwebp_webpdemux_LINK_LIBS_RELWITHDEBINFO}
             ${libwebp_libwebp_webpdemux_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET libwebp::webpdemux PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${libwebp_libwebp_webpdemux_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET libwebp::webpdemux PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${libwebp_libwebp_webpdemux_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET libwebp::webpdemux PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${libwebp_libwebp_webpdemux_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET libwebp::webpdemux PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${libwebp_libwebp_webpdemux_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${libwebp_libwebp_webpdemux_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(libwebp_libwebp_webpdemux_TARGET_PROPERTIES TRUE)
########## COMPONENT libwebp::webp TARGET PROPERTIES ######################################
set_property(TARGET libwebp::webp PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${libwebp_libwebp_webp_LINK_LIBS_RELWITHDEBINFO}
             ${libwebp_libwebp_webp_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET libwebp::webp PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${libwebp_libwebp_webp_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET libwebp::webp PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${libwebp_libwebp_webp_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET libwebp::webp PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${libwebp_libwebp_webp_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET libwebp::webp PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${libwebp_libwebp_webp_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${libwebp_libwebp_webp_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(libwebp_libwebp_webp_TARGET_PROPERTIES TRUE)
########## COMPONENT libwebp::webpdecoder TARGET PROPERTIES ######################################
set_property(TARGET libwebp::webpdecoder PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${libwebp_libwebp_webpdecoder_LINK_LIBS_RELWITHDEBINFO}
             ${libwebp_libwebp_webpdecoder_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET libwebp::webpdecoder PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${libwebp_libwebp_webpdecoder_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET libwebp::webpdecoder PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${libwebp_libwebp_webpdecoder_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET libwebp::webpdecoder PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${libwebp_libwebp_webpdecoder_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET libwebp::webpdecoder PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${libwebp_libwebp_webpdecoder_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${libwebp_libwebp_webpdecoder_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(libwebp_libwebp_webpdecoder_TARGET_PROPERTIES TRUE)