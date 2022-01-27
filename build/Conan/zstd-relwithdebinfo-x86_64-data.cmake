########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(zstd_COMPONENT_NAMES ${zstd_COMPONENT_NAMES} zstd::libzstd_static)
list(REMOVE_DUPLICATES zstd_COMPONENT_NAMES)
set(zstd_FIND_DEPENDENCY_NAMES ${zstd_FIND_DEPENDENCY_NAMES} )
list(REMOVE_DUPLICATES zstd_FIND_DEPENDENCY_NAMES)

########### VARIABLES #######################################################################
#############################################################################################
set(zstd_PACKAGE_FOLDER_RELWITHDEBINFO "/home/kudan/.conan/data/zstd/1.5.1/_/_/package/6b7ff26bfd4c2cf2ccba522bfba2d2e7820e40da")
set(zstd_INCLUDE_DIRS_RELWITHDEBINFO "${zstd_PACKAGE_FOLDER_RELWITHDEBINFO}/include")
set(zstd_RES_DIRS_RELWITHDEBINFO "${zstd_PACKAGE_FOLDER_RELWITHDEBINFO}/res")
set(zstd_DEFINITIONS_RELWITHDEBINFO )
set(zstd_SHARED_LINK_FLAGS_RELWITHDEBINFO )
set(zstd_EXE_LINK_FLAGS_RELWITHDEBINFO )
set(zstd_OBJECTS_RELWITHDEBINFO )
set(zstd_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(zstd_COMPILE_OPTIONS_C_RELWITHDEBINFO )
set(zstd_COMPILE_OPTIONS_CXX_RELWITHDEBINFO )
set(zstd_LIB_DIRS_RELWITHDEBINFO "${zstd_PACKAGE_FOLDER_RELWITHDEBINFO}/lib")
set(zstd_LIBS_RELWITHDEBINFO zstd)
set(zstd_SYSTEM_LIBS_RELWITHDEBINFO pthread)
set(zstd_FRAMEWORK_DIRS_RELWITHDEBINFO "${zstd_PACKAGE_FOLDER_RELWITHDEBINFO}/Frameworks")
set(zstd_FRAMEWORKS_RELWITHDEBINFO )
set(zstd_BUILD_MODULES_PATHS_RELWITHDEBINFO )
set(zstd_BUILD_DIRS_RELWITHDEBINFO "${zstd_PACKAGE_FOLDER_RELWITHDEBINFO}/")

set(zstd_COMPONENTS_RELWITHDEBINFO zstd::libzstd_static)
########### COMPONENT zstd::libzstd_static VARIABLES #############################################
set(zstd_zstd_libzstd_static_INCLUDE_DIRS_RELWITHDEBINFO "${zstd_PACKAGE_FOLDER_RELWITHDEBINFO}/include")
set(zstd_zstd_libzstd_static_LIB_DIRS_RELWITHDEBINFO "${zstd_PACKAGE_FOLDER_RELWITHDEBINFO}/lib")
set(zstd_zstd_libzstd_static_RES_DIRS_RELWITHDEBINFO "${zstd_PACKAGE_FOLDER_RELWITHDEBINFO}/res")
set(zstd_zstd_libzstd_static_DEFINITIONS_RELWITHDEBINFO )
set(zstd_zstd_libzstd_static_OBJECTS_RELWITHDEBINFO )
set(zstd_zstd_libzstd_static_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(zstd_zstd_libzstd_static_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(zstd_zstd_libzstd_static_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(zstd_zstd_libzstd_static_LIBS_RELWITHDEBINFO zstd)
set(zstd_zstd_libzstd_static_SYSTEM_LIBS_RELWITHDEBINFO pthread)
set(zstd_zstd_libzstd_static_FRAMEWORK_DIRS_RELWITHDEBINFO "${zstd_PACKAGE_FOLDER_RELWITHDEBINFO}/Frameworks")
set(zstd_zstd_libzstd_static_FRAMEWORKS_RELWITHDEBINFO )
set(zstd_zstd_libzstd_static_DEPENDENCIES_RELWITHDEBINFO )
set(zstd_zstd_libzstd_static_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)