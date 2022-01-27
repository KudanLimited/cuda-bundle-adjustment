########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(libwebp_COMPONENT_NAMES ${libwebp_COMPONENT_NAMES} libwebp::webpdecoder libwebp::webp libwebp::webpdemux libwebp::webpmux)
list(REMOVE_DUPLICATES libwebp_COMPONENT_NAMES)
set(libwebp_FIND_DEPENDENCY_NAMES ${libwebp_FIND_DEPENDENCY_NAMES} )
list(REMOVE_DUPLICATES libwebp_FIND_DEPENDENCY_NAMES)

########### VARIABLES #######################################################################
#############################################################################################
set(libwebp_PACKAGE_FOLDER_RELWITHDEBINFO "/home/kudan/.conan/data/libwebp/1.2.0/_/_/package/701d9c6d6d4798bc8d444813c228ba745babc496")
set(libwebp_INCLUDE_DIRS_RELWITHDEBINFO "${libwebp_PACKAGE_FOLDER_RELWITHDEBINFO}/include")
set(libwebp_RES_DIRS_RELWITHDEBINFO "${libwebp_PACKAGE_FOLDER_RELWITHDEBINFO}/res")
set(libwebp_DEFINITIONS_RELWITHDEBINFO )
set(libwebp_SHARED_LINK_FLAGS_RELWITHDEBINFO )
set(libwebp_EXE_LINK_FLAGS_RELWITHDEBINFO )
set(libwebp_OBJECTS_RELWITHDEBINFO )
set(libwebp_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(libwebp_COMPILE_OPTIONS_C_RELWITHDEBINFO )
set(libwebp_COMPILE_OPTIONS_CXX_RELWITHDEBINFO )
set(libwebp_LIB_DIRS_RELWITHDEBINFO "${libwebp_PACKAGE_FOLDER_RELWITHDEBINFO}/lib")
set(libwebp_LIBS_RELWITHDEBINFO webpmux webpdemux webp webpdecoder)
set(libwebp_SYSTEM_LIBS_RELWITHDEBINFO m pthread)
set(libwebp_FRAMEWORK_DIRS_RELWITHDEBINFO "${libwebp_PACKAGE_FOLDER_RELWITHDEBINFO}/Frameworks")
set(libwebp_FRAMEWORKS_RELWITHDEBINFO )
set(libwebp_BUILD_MODULES_PATHS_RELWITHDEBINFO )
set(libwebp_BUILD_DIRS_RELWITHDEBINFO "${libwebp_PACKAGE_FOLDER_RELWITHDEBINFO}/")

set(libwebp_COMPONENTS_RELWITHDEBINFO libwebp::webpdecoder libwebp::webp libwebp::webpdemux libwebp::webpmux)
########### COMPONENT libwebp::webpmux VARIABLES #############################################
set(libwebp_libwebp_webpmux_INCLUDE_DIRS_RELWITHDEBINFO "${libwebp_PACKAGE_FOLDER_RELWITHDEBINFO}/include")
set(libwebp_libwebp_webpmux_LIB_DIRS_RELWITHDEBINFO "${libwebp_PACKAGE_FOLDER_RELWITHDEBINFO}/lib")
set(libwebp_libwebp_webpmux_RES_DIRS_RELWITHDEBINFO "${libwebp_PACKAGE_FOLDER_RELWITHDEBINFO}/res")
set(libwebp_libwebp_webpmux_DEFINITIONS_RELWITHDEBINFO )
set(libwebp_libwebp_webpmux_OBJECTS_RELWITHDEBINFO )
set(libwebp_libwebp_webpmux_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(libwebp_libwebp_webpmux_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(libwebp_libwebp_webpmux_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(libwebp_libwebp_webpmux_LIBS_RELWITHDEBINFO webpmux)
set(libwebp_libwebp_webpmux_SYSTEM_LIBS_RELWITHDEBINFO m)
set(libwebp_libwebp_webpmux_FRAMEWORK_DIRS_RELWITHDEBINFO "${libwebp_PACKAGE_FOLDER_RELWITHDEBINFO}/Frameworks")
set(libwebp_libwebp_webpmux_FRAMEWORKS_RELWITHDEBINFO )
set(libwebp_libwebp_webpmux_DEPENDENCIES_RELWITHDEBINFO libwebp::webp)
set(libwebp_libwebp_webpmux_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT libwebp::webpdemux VARIABLES #############################################
set(libwebp_libwebp_webpdemux_INCLUDE_DIRS_RELWITHDEBINFO "${libwebp_PACKAGE_FOLDER_RELWITHDEBINFO}/include")
set(libwebp_libwebp_webpdemux_LIB_DIRS_RELWITHDEBINFO "${libwebp_PACKAGE_FOLDER_RELWITHDEBINFO}/lib")
set(libwebp_libwebp_webpdemux_RES_DIRS_RELWITHDEBINFO "${libwebp_PACKAGE_FOLDER_RELWITHDEBINFO}/res")
set(libwebp_libwebp_webpdemux_DEFINITIONS_RELWITHDEBINFO )
set(libwebp_libwebp_webpdemux_OBJECTS_RELWITHDEBINFO )
set(libwebp_libwebp_webpdemux_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(libwebp_libwebp_webpdemux_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(libwebp_libwebp_webpdemux_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(libwebp_libwebp_webpdemux_LIBS_RELWITHDEBINFO webpdemux)
set(libwebp_libwebp_webpdemux_SYSTEM_LIBS_RELWITHDEBINFO )
set(libwebp_libwebp_webpdemux_FRAMEWORK_DIRS_RELWITHDEBINFO "${libwebp_PACKAGE_FOLDER_RELWITHDEBINFO}/Frameworks")
set(libwebp_libwebp_webpdemux_FRAMEWORKS_RELWITHDEBINFO )
set(libwebp_libwebp_webpdemux_DEPENDENCIES_RELWITHDEBINFO libwebp::webp)
set(libwebp_libwebp_webpdemux_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT libwebp::webp VARIABLES #############################################
set(libwebp_libwebp_webp_INCLUDE_DIRS_RELWITHDEBINFO "${libwebp_PACKAGE_FOLDER_RELWITHDEBINFO}/include")
set(libwebp_libwebp_webp_LIB_DIRS_RELWITHDEBINFO "${libwebp_PACKAGE_FOLDER_RELWITHDEBINFO}/lib")
set(libwebp_libwebp_webp_RES_DIRS_RELWITHDEBINFO "${libwebp_PACKAGE_FOLDER_RELWITHDEBINFO}/res")
set(libwebp_libwebp_webp_DEFINITIONS_RELWITHDEBINFO )
set(libwebp_libwebp_webp_OBJECTS_RELWITHDEBINFO )
set(libwebp_libwebp_webp_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(libwebp_libwebp_webp_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(libwebp_libwebp_webp_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(libwebp_libwebp_webp_LIBS_RELWITHDEBINFO webp)
set(libwebp_libwebp_webp_SYSTEM_LIBS_RELWITHDEBINFO m pthread)
set(libwebp_libwebp_webp_FRAMEWORK_DIRS_RELWITHDEBINFO "${libwebp_PACKAGE_FOLDER_RELWITHDEBINFO}/Frameworks")
set(libwebp_libwebp_webp_FRAMEWORKS_RELWITHDEBINFO )
set(libwebp_libwebp_webp_DEPENDENCIES_RELWITHDEBINFO )
set(libwebp_libwebp_webp_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT libwebp::webpdecoder VARIABLES #############################################
set(libwebp_libwebp_webpdecoder_INCLUDE_DIRS_RELWITHDEBINFO "${libwebp_PACKAGE_FOLDER_RELWITHDEBINFO}/include")
set(libwebp_libwebp_webpdecoder_LIB_DIRS_RELWITHDEBINFO "${libwebp_PACKAGE_FOLDER_RELWITHDEBINFO}/lib")
set(libwebp_libwebp_webpdecoder_RES_DIRS_RELWITHDEBINFO "${libwebp_PACKAGE_FOLDER_RELWITHDEBINFO}/res")
set(libwebp_libwebp_webpdecoder_DEFINITIONS_RELWITHDEBINFO )
set(libwebp_libwebp_webpdecoder_OBJECTS_RELWITHDEBINFO )
set(libwebp_libwebp_webpdecoder_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(libwebp_libwebp_webpdecoder_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(libwebp_libwebp_webpdecoder_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(libwebp_libwebp_webpdecoder_LIBS_RELWITHDEBINFO webpdecoder)
set(libwebp_libwebp_webpdecoder_SYSTEM_LIBS_RELWITHDEBINFO pthread)
set(libwebp_libwebp_webpdecoder_FRAMEWORK_DIRS_RELWITHDEBINFO "${libwebp_PACKAGE_FOLDER_RELWITHDEBINFO}/Frameworks")
set(libwebp_libwebp_webpdecoder_FRAMEWORKS_RELWITHDEBINFO )
set(libwebp_libwebp_webpdecoder_DEPENDENCIES_RELWITHDEBINFO )
set(libwebp_libwebp_webpdecoder_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)