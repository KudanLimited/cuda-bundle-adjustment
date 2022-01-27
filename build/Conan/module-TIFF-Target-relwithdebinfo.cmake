########### VARIABLES #######################################################################
#############################################################################################

set(libtiff_COMPILE_OPTIONS_RELWITHDEBINFO
    "$<$<COMPILE_LANGUAGE:CXX>:${libtiff_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>"
    "$<$<COMPILE_LANGUAGE:C>:${libtiff_COMPILE_OPTIONS_C_RELWITHDEBINFO}>")

set(libtiff_LINKER_FLAGS_RELWITHDEBINFO
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${libtiff_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${libtiff_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${libtiff_EXE_LINK_FLAGS_RELWITHDEBINFO}>")

set(libtiff_FRAMEWORKS_FOUND_RELWITHDEBINFO "") # Will be filled later
conan_find_apple_frameworks(libtiff_FRAMEWORKS_FOUND_RELWITHDEBINFO "${libtiff_FRAMEWORKS_RELWITHDEBINFO}" "${libtiff_FRAMEWORK_DIRS_RELWITHDEBINFO}")

# Gather all the libraries that should be linked to the targets (do not touch existing variables)
set(_libtiff_DEPENDENCIES_RELWITHDEBINFO "${libtiff_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${libtiff_SYSTEM_LIBS_RELWITHDEBINFO} ZLIB::ZLIB;libdeflate::libdeflate;LibLZMA::LibLZMA;JPEG::JPEG;JBIG::JBIG;zstd::libzstd_static;libwebp::libwebp")

set(libtiff_LIBRARIES_TARGETS_RELWITHDEBINFO "") # Will be filled later
set(libtiff_LIBRARIES_RELWITHDEBINFO "") # Will be filled later
conan_package_library_targets("${libtiff_LIBS_RELWITHDEBINFO}"    # libraries
                              "${libtiff_LIB_DIRS_RELWITHDEBINFO}" # package_libdir
                              "${_libtiff_DEPENDENCIES_RELWITHDEBINFO}" # deps
                              libtiff_LIBRARIES_RELWITHDEBINFO   # out_libraries
                              libtiff_LIBRARIES_TARGETS_RELWITHDEBINFO  # out_libraries_targets
                              "_RELWITHDEBINFO"  # config_suffix
                              "libtiff")    # package_name

foreach(_FRAMEWORK ${libtiff_FRAMEWORKS_FOUND_RELWITHDEBINFO})
    list(APPEND libtiff_LIBRARIES_TARGETS_RELWITHDEBINFO ${_FRAMEWORK})
    list(APPEND libtiff_LIBRARIES_RELWITHDEBINFO ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${libtiff_SYSTEM_LIBS_RELWITHDEBINFO})
    list(APPEND libtiff_LIBRARIES_TARGETS_RELWITHDEBINFO ${_SYSTEM_LIB})
    list(APPEND libtiff_LIBRARIES_RELWITHDEBINFO ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(libtiff_LIBRARIES_TARGETS_RELWITHDEBINFO "${libtiff_LIBRARIES_TARGETS_RELWITHDEBINFO};ZLIB::ZLIB;libdeflate::libdeflate;LibLZMA::LibLZMA;JPEG::JPEG;JBIG::JBIG;zstd::libzstd_static;libwebp::libwebp")
set(libtiff_LIBRARIES_RELWITHDEBINFO "${libtiff_LIBRARIES_RELWITHDEBINFO};ZLIB::ZLIB;libdeflate::libdeflate;LibLZMA::LibLZMA;JPEG::JPEG;JBIG::JBIG;zstd::libzstd_static;libwebp::libwebp")

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${libtiff_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH ${libtiff_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_PREFIX_PATH})


########## GLOBAL TARGET PROPERTIES RelWithDebInfo ########################################
set_property(TARGET TIFF::TIFF
             PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${libtiff_LIBRARIES_TARGETS_RELWITHDEBINFO}
                                           ${libtiff_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET TIFF::TIFF
             PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${libtiff_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET TIFF::TIFF
             PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${libtiff_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET TIFF::TIFF
             PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${libtiff_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET TIFF::TIFF
             PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${libtiff_COMPILE_OPTIONS_RELWITHDEBINFO}> APPEND)

########## COMPONENTS TARGET PROPERTIES RelWithDebInfo ########################################