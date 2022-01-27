########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(libtiff_COMPONENT_NAMES ${libtiff_COMPONENT_NAMES} )
list(REMOVE_DUPLICATES libtiff_COMPONENT_NAMES)
set(libtiff_FIND_DEPENDENCY_NAMES ${libtiff_FIND_DEPENDENCY_NAMES} ZLIB libdeflate LibLZMA libjpeg JBIG zstd libwebp)
list(REMOVE_DUPLICATES libtiff_FIND_DEPENDENCY_NAMES)

########### VARIABLES #######################################################################
#############################################################################################
set(libtiff_PACKAGE_FOLDER_RELWITHDEBINFO "/home/kudan/.conan/data/libtiff/4.3.0/_/_/package/9bdae2649e516367a6bbbaaf5857a150ae54c8d3")
set(libtiff_INCLUDE_DIRS_RELWITHDEBINFO "${libtiff_PACKAGE_FOLDER_RELWITHDEBINFO}/include")
set(libtiff_RES_DIRS_RELWITHDEBINFO "${libtiff_PACKAGE_FOLDER_RELWITHDEBINFO}/res")
set(libtiff_DEFINITIONS_RELWITHDEBINFO )
set(libtiff_SHARED_LINK_FLAGS_RELWITHDEBINFO )
set(libtiff_EXE_LINK_FLAGS_RELWITHDEBINFO )
set(libtiff_OBJECTS_RELWITHDEBINFO )
set(libtiff_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(libtiff_COMPILE_OPTIONS_C_RELWITHDEBINFO )
set(libtiff_COMPILE_OPTIONS_CXX_RELWITHDEBINFO )
set(libtiff_LIB_DIRS_RELWITHDEBINFO "${libtiff_PACKAGE_FOLDER_RELWITHDEBINFO}/lib")
set(libtiff_LIBS_RELWITHDEBINFO tiffxx tiff)
set(libtiff_SYSTEM_LIBS_RELWITHDEBINFO m)
set(libtiff_FRAMEWORK_DIRS_RELWITHDEBINFO "${libtiff_PACKAGE_FOLDER_RELWITHDEBINFO}/Frameworks")
set(libtiff_FRAMEWORKS_RELWITHDEBINFO )
set(libtiff_BUILD_MODULES_PATHS_RELWITHDEBINFO )
set(libtiff_BUILD_DIRS_RELWITHDEBINFO "${libtiff_PACKAGE_FOLDER_RELWITHDEBINFO}/")

set(libtiff_COMPONENTS_RELWITHDEBINFO )