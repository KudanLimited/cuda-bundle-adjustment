########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(opencv_COMPONENT_NAMES ${opencv_COMPONENT_NAMES} opencv::opencv_core opencv::opencv_core_alias opencv::opencv_flann opencv::opencv_flann_alias opencv::opencv_imgproc opencv::opencv_imgproc_alias opencv::opencv_ml opencv::opencv_ml_alias opencv::opencv_photo opencv::opencv_photo_alias opencv::opencv_features2d opencv::opencv_features2d_alias opencv::opencv_imgcodecs opencv::opencv_imgcodecs_alias opencv::opencv_videoio opencv::opencv_videoio_alias opencv::opencv_calib3d opencv::opencv_calib3d_alias opencv::opencv_objdetect opencv::opencv_objdetect_alias opencv::opencv_stitching opencv::opencv_stitching_alias opencv::opencv_video opencv::opencv_video_alias opencv::opencv_highgui opencv::opencv_highgui_alias)
list(REMOVE_DUPLICATES opencv_COMPONENT_NAMES)
set(opencv_FIND_DEPENDENCY_NAMES ${opencv_FIND_DEPENDENCY_NAMES} zlib libpng JPEG tiff libwebp jasper JPEG tiff JPEG tiff)
list(REMOVE_DUPLICATES opencv_FIND_DEPENDENCY_NAMES)

########### VARIABLES #######################################################################
#############################################################################################
set(opencv_PACKAGE_FOLDER_RELWITHDEBINFO "/home/kudan/.conan/data/opencv/4.0.1/_/_/package/8c33e7eafe6a9e7de92282766132db65ff6b6cc9")
set(opencv_INCLUDE_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/include")
set(opencv_RES_DIRS_RELWITHDEBINFO )
set(opencv_DEFINITIONS_RELWITHDEBINFO )
set(opencv_SHARED_LINK_FLAGS_RELWITHDEBINFO )
set(opencv_EXE_LINK_FLAGS_RELWITHDEBINFO )
set(opencv_OBJECTS_RELWITHDEBINFO )
set(opencv_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_COMPILE_OPTIONS_C_RELWITHDEBINFO )
set(opencv_COMPILE_OPTIONS_CXX_RELWITHDEBINFO )
set(opencv_LIB_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/lib")
set(opencv_LIBS_RELWITHDEBINFO opencv_highgui opencv_video opencv_stitching opencv_objdetect opencv_calib3d opencv_videoio opencv_imgcodecs opencv_features2d opencv_photo opencv_ml opencv_imgproc opencv_flann opencv_core)
set(opencv_SYSTEM_LIBS_RELWITHDEBINFO dl m pthread rt)
set(opencv_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_BUILD_MODULES_PATHS_RELWITHDEBINFO )
set(opencv_BUILD_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/lib/cmake")

set(opencv_COMPONENTS_RELWITHDEBINFO opencv::opencv_core opencv::opencv_core_alias opencv::opencv_flann opencv::opencv_flann_alias opencv::opencv_imgproc opencv::opencv_imgproc_alias opencv::opencv_ml opencv::opencv_ml_alias opencv::opencv_photo opencv::opencv_photo_alias opencv::opencv_features2d opencv::opencv_features2d_alias opencv::opencv_imgcodecs opencv::opencv_imgcodecs_alias opencv::opencv_videoio opencv::opencv_videoio_alias opencv::opencv_calib3d opencv::opencv_calib3d_alias opencv::opencv_objdetect opencv::opencv_objdetect_alias opencv::opencv_stitching opencv::opencv_stitching_alias opencv::opencv_video opencv::opencv_video_alias opencv::opencv_highgui opencv::opencv_highgui_alias)
########### COMPONENT opencv::opencv_highgui_alias VARIABLES #############################################
set(opencv_opencv_opencv_highgui_alias_INCLUDE_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_highgui_alias_LIB_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_highgui_alias_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_highgui_alias_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_highgui_alias_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_highgui_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_highgui_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_highgui_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_highgui_alias_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_highgui_alias_SYSTEM_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_highgui_alias_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_highgui_alias_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_highgui_alias_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_highgui)
set(opencv_opencv_opencv_highgui_alias_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_highgui VARIABLES #############################################
set(opencv_opencv_opencv_highgui_INCLUDE_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/include")
set(opencv_opencv_opencv_highgui_LIB_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/lib")
set(opencv_opencv_opencv_highgui_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_highgui_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_highgui_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_highgui_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_highgui_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_highgui_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_highgui_LIBS_RELWITHDEBINFO opencv_highgui)
set(opencv_opencv_opencv_highgui_SYSTEM_LIBS_RELWITHDEBINFO dl m pthread rt)
set(opencv_opencv_opencv_highgui_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_highgui_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_highgui_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_core opencv::opencv_imgproc opencv::opencv_imgcodecs opencv::opencv_videoio)
set(opencv_opencv_opencv_highgui_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_video_alias VARIABLES #############################################
set(opencv_opencv_opencv_video_alias_INCLUDE_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_video_alias_LIB_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_video_alias_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_video_alias_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_video_alias_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_video_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_video_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_video_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_video_alias_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_video_alias_SYSTEM_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_video_alias_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_video_alias_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_video_alias_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_video)
set(opencv_opencv_opencv_video_alias_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_video VARIABLES #############################################
set(opencv_opencv_opencv_video_INCLUDE_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/include")
set(opencv_opencv_opencv_video_LIB_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/lib")
set(opencv_opencv_opencv_video_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_video_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_video_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_video_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_video_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_video_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_video_LIBS_RELWITHDEBINFO opencv_video)
set(opencv_opencv_opencv_video_SYSTEM_LIBS_RELWITHDEBINFO dl m pthread rt)
set(opencv_opencv_opencv_video_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_video_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_video_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_core opencv::opencv_flann opencv::opencv_imgproc opencv::opencv_features2d opencv::opencv_calib3d)
set(opencv_opencv_opencv_video_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_stitching_alias VARIABLES #############################################
set(opencv_opencv_opencv_stitching_alias_INCLUDE_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_stitching_alias_LIB_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_stitching_alias_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_stitching_alias_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_stitching_alias_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_stitching_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_stitching_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_stitching_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_stitching_alias_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_stitching_alias_SYSTEM_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_stitching_alias_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_stitching_alias_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_stitching_alias_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_stitching)
set(opencv_opencv_opencv_stitching_alias_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_stitching VARIABLES #############################################
set(opencv_opencv_opencv_stitching_INCLUDE_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/include")
set(opencv_opencv_opencv_stitching_LIB_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/lib")
set(opencv_opencv_opencv_stitching_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_stitching_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_stitching_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_stitching_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_stitching_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_stitching_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_stitching_LIBS_RELWITHDEBINFO opencv_stitching)
set(opencv_opencv_opencv_stitching_SYSTEM_LIBS_RELWITHDEBINFO dl m pthread rt)
set(opencv_opencv_opencv_stitching_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_stitching_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_stitching_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_core opencv::opencv_flann opencv::opencv_imgproc opencv::opencv_features2d opencv::opencv_calib3d)
set(opencv_opencv_opencv_stitching_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_objdetect_alias VARIABLES #############################################
set(opencv_opencv_opencv_objdetect_alias_INCLUDE_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_objdetect_alias_LIB_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_objdetect_alias_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_objdetect_alias_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_objdetect_alias_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_objdetect_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_objdetect_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_objdetect_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_objdetect_alias_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_objdetect_alias_SYSTEM_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_objdetect_alias_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_objdetect_alias_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_objdetect_alias_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_objdetect)
set(opencv_opencv_opencv_objdetect_alias_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_objdetect VARIABLES #############################################
set(opencv_opencv_opencv_objdetect_INCLUDE_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/include")
set(opencv_opencv_opencv_objdetect_LIB_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/lib")
set(opencv_opencv_opencv_objdetect_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_objdetect_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_objdetect_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_objdetect_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_objdetect_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_objdetect_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_objdetect_LIBS_RELWITHDEBINFO opencv_objdetect)
set(opencv_opencv_opencv_objdetect_SYSTEM_LIBS_RELWITHDEBINFO dl m pthread rt)
set(opencv_opencv_opencv_objdetect_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_objdetect_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_objdetect_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_core opencv::opencv_flann opencv::opencv_imgproc opencv::opencv_features2d opencv::opencv_calib3d)
set(opencv_opencv_opencv_objdetect_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_calib3d_alias VARIABLES #############################################
set(opencv_opencv_opencv_calib3d_alias_INCLUDE_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_calib3d_alias_LIB_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_calib3d_alias_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_calib3d_alias_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_calib3d_alias_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_calib3d_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_calib3d_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_calib3d_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_calib3d_alias_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_calib3d_alias_SYSTEM_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_calib3d_alias_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_calib3d_alias_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_calib3d_alias_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_calib3d)
set(opencv_opencv_opencv_calib3d_alias_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_calib3d VARIABLES #############################################
set(opencv_opencv_opencv_calib3d_INCLUDE_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/include")
set(opencv_opencv_opencv_calib3d_LIB_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/lib")
set(opencv_opencv_opencv_calib3d_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_calib3d_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_calib3d_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_calib3d_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_calib3d_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_calib3d_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_calib3d_LIBS_RELWITHDEBINFO opencv_calib3d)
set(opencv_opencv_opencv_calib3d_SYSTEM_LIBS_RELWITHDEBINFO dl m pthread rt)
set(opencv_opencv_opencv_calib3d_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_calib3d_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_calib3d_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_core opencv::opencv_flann opencv::opencv_imgproc opencv::opencv_features2d)
set(opencv_opencv_opencv_calib3d_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_videoio_alias VARIABLES #############################################
set(opencv_opencv_opencv_videoio_alias_INCLUDE_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_videoio_alias_LIB_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_videoio_alias_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_videoio_alias_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_videoio_alias_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_videoio_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_videoio_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_videoio_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_videoio_alias_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_videoio_alias_SYSTEM_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_videoio_alias_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_videoio_alias_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_videoio_alias_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_videoio)
set(opencv_opencv_opencv_videoio_alias_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_videoio VARIABLES #############################################
set(opencv_opencv_opencv_videoio_INCLUDE_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/include")
set(opencv_opencv_opencv_videoio_LIB_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/lib")
set(opencv_opencv_opencv_videoio_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_videoio_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_videoio_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_videoio_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_videoio_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_videoio_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_videoio_LIBS_RELWITHDEBINFO opencv_videoio)
set(opencv_opencv_opencv_videoio_SYSTEM_LIBS_RELWITHDEBINFO dl m pthread rt)
set(opencv_opencv_opencv_videoio_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_videoio_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_videoio_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_core opencv::opencv_imgproc opencv::opencv_imgcodecs)
set(opencv_opencv_opencv_videoio_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_imgcodecs_alias VARIABLES #############################################
set(opencv_opencv_opencv_imgcodecs_alias_INCLUDE_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgcodecs_alias_LIB_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgcodecs_alias_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgcodecs_alias_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgcodecs_alias_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgcodecs_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgcodecs_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_imgcodecs_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_imgcodecs_alias_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgcodecs_alias_SYSTEM_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgcodecs_alias_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgcodecs_alias_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgcodecs_alias_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_imgcodecs)
set(opencv_opencv_opencv_imgcodecs_alias_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_imgcodecs VARIABLES #############################################
set(opencv_opencv_opencv_imgcodecs_INCLUDE_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/include")
set(opencv_opencv_opencv_imgcodecs_LIB_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/lib")
set(opencv_opencv_opencv_imgcodecs_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgcodecs_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgcodecs_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgcodecs_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgcodecs_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_imgcodecs_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_imgcodecs_LIBS_RELWITHDEBINFO opencv_imgcodecs)
set(opencv_opencv_opencv_imgcodecs_SYSTEM_LIBS_RELWITHDEBINFO dl m pthread rt)
set(opencv_opencv_opencv_imgcodecs_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgcodecs_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgcodecs_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_core opencv::opencv_imgproc ZLIB::ZLIB PNG::PNG JPEG::JPEG TIFF::TIFF libwebp::libwebp Jasper::Jasper)
set(opencv_opencv_opencv_imgcodecs_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_features2d_alias VARIABLES #############################################
set(opencv_opencv_opencv_features2d_alias_INCLUDE_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_features2d_alias_LIB_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_features2d_alias_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_features2d_alias_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_features2d_alias_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_features2d_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_features2d_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_features2d_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_features2d_alias_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_features2d_alias_SYSTEM_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_features2d_alias_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_features2d_alias_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_features2d_alias_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_features2d)
set(opencv_opencv_opencv_features2d_alias_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_features2d VARIABLES #############################################
set(opencv_opencv_opencv_features2d_INCLUDE_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/include")
set(opencv_opencv_opencv_features2d_LIB_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/lib")
set(opencv_opencv_opencv_features2d_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_features2d_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_features2d_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_features2d_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_features2d_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_features2d_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_features2d_LIBS_RELWITHDEBINFO opencv_features2d)
set(opencv_opencv_opencv_features2d_SYSTEM_LIBS_RELWITHDEBINFO dl m pthread rt)
set(opencv_opencv_opencv_features2d_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_features2d_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_features2d_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_core opencv::opencv_flann opencv::opencv_imgproc Jasper::Jasper)
set(opencv_opencv_opencv_features2d_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_photo_alias VARIABLES #############################################
set(opencv_opencv_opencv_photo_alias_INCLUDE_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_photo_alias_LIB_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_photo_alias_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_photo_alias_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_photo_alias_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_photo_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_photo_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_photo_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_photo_alias_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_photo_alias_SYSTEM_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_photo_alias_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_photo_alias_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_photo_alias_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_photo)
set(opencv_opencv_opencv_photo_alias_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_photo VARIABLES #############################################
set(opencv_opencv_opencv_photo_INCLUDE_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/include")
set(opencv_opencv_opencv_photo_LIB_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/lib")
set(opencv_opencv_opencv_photo_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_photo_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_photo_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_photo_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_photo_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_photo_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_photo_LIBS_RELWITHDEBINFO opencv_photo)
set(opencv_opencv_opencv_photo_SYSTEM_LIBS_RELWITHDEBINFO dl m pthread rt)
set(opencv_opencv_opencv_photo_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_photo_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_photo_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_core opencv::opencv_imgproc)
set(opencv_opencv_opencv_photo_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_ml_alias VARIABLES #############################################
set(opencv_opencv_opencv_ml_alias_INCLUDE_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_ml_alias_LIB_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_ml_alias_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_ml_alias_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_ml_alias_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_ml_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_ml_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_ml_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_ml_alias_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_ml_alias_SYSTEM_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_ml_alias_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_ml_alias_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_ml_alias_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_ml)
set(opencv_opencv_opencv_ml_alias_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_ml VARIABLES #############################################
set(opencv_opencv_opencv_ml_INCLUDE_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/include")
set(opencv_opencv_opencv_ml_LIB_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/lib")
set(opencv_opencv_opencv_ml_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_ml_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_ml_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_ml_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_ml_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_ml_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_ml_LIBS_RELWITHDEBINFO opencv_ml)
set(opencv_opencv_opencv_ml_SYSTEM_LIBS_RELWITHDEBINFO dl m pthread rt)
set(opencv_opencv_opencv_ml_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_ml_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_ml_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_core)
set(opencv_opencv_opencv_ml_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_imgproc_alias VARIABLES #############################################
set(opencv_opencv_opencv_imgproc_alias_INCLUDE_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgproc_alias_LIB_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgproc_alias_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgproc_alias_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgproc_alias_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgproc_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgproc_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_imgproc_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_imgproc_alias_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgproc_alias_SYSTEM_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgproc_alias_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgproc_alias_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgproc_alias_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_imgproc)
set(opencv_opencv_opencv_imgproc_alias_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_imgproc VARIABLES #############################################
set(opencv_opencv_opencv_imgproc_INCLUDE_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/include")
set(opencv_opencv_opencv_imgproc_LIB_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/lib")
set(opencv_opencv_opencv_imgproc_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgproc_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgproc_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgproc_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgproc_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_imgproc_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_imgproc_LIBS_RELWITHDEBINFO opencv_imgproc)
set(opencv_opencv_opencv_imgproc_SYSTEM_LIBS_RELWITHDEBINFO dl m pthread rt)
set(opencv_opencv_opencv_imgproc_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgproc_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_imgproc_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_core)
set(opencv_opencv_opencv_imgproc_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_flann_alias VARIABLES #############################################
set(opencv_opencv_opencv_flann_alias_INCLUDE_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_flann_alias_LIB_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_flann_alias_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_flann_alias_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_flann_alias_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_flann_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_flann_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_flann_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_flann_alias_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_flann_alias_SYSTEM_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_flann_alias_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_flann_alias_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_flann_alias_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_flann)
set(opencv_opencv_opencv_flann_alias_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_flann VARIABLES #############################################
set(opencv_opencv_opencv_flann_INCLUDE_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/include")
set(opencv_opencv_opencv_flann_LIB_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/lib")
set(opencv_opencv_opencv_flann_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_flann_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_flann_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_flann_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_flann_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_flann_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_flann_LIBS_RELWITHDEBINFO opencv_flann)
set(opencv_opencv_opencv_flann_SYSTEM_LIBS_RELWITHDEBINFO dl m pthread rt)
set(opencv_opencv_opencv_flann_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_flann_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_flann_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_core)
set(opencv_opencv_opencv_flann_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_core_alias VARIABLES #############################################
set(opencv_opencv_opencv_core_alias_INCLUDE_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_core_alias_LIB_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_core_alias_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_core_alias_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_core_alias_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_core_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_core_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_core_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_core_alias_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_core_alias_SYSTEM_LIBS_RELWITHDEBINFO )
set(opencv_opencv_opencv_core_alias_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_core_alias_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_core_alias_DEPENDENCIES_RELWITHDEBINFO opencv::opencv_core)
set(opencv_opencv_opencv_core_alias_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)
########### COMPONENT opencv::opencv_core VARIABLES #############################################
set(opencv_opencv_opencv_core_INCLUDE_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/include")
set(opencv_opencv_opencv_core_LIB_DIRS_RELWITHDEBINFO "${opencv_PACKAGE_FOLDER_RELWITHDEBINFO}/lib")
set(opencv_opencv_opencv_core_RES_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_core_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_core_OBJECTS_RELWITHDEBINFO )
set(opencv_opencv_opencv_core_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(opencv_opencv_opencv_core_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(opencv_opencv_opencv_core_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(opencv_opencv_opencv_core_LIBS_RELWITHDEBINFO opencv_core)
set(opencv_opencv_opencv_core_SYSTEM_LIBS_RELWITHDEBINFO dl m pthread rt)
set(opencv_opencv_opencv_core_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(opencv_opencv_opencv_core_FRAMEWORKS_RELWITHDEBINFO )
set(opencv_opencv_opencv_core_DEPENDENCIES_RELWITHDEBINFO ZLIB::ZLIB)
set(opencv_opencv_opencv_core_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)