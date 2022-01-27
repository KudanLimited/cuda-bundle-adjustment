########### VARIABLES #######################################################################
#############################################################################################

set(opencv_COMPILE_OPTIONS_RELWITHDEBINFO
    "$<$<COMPILE_LANGUAGE:CXX>:${opencv_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>"
    "$<$<COMPILE_LANGUAGE:C>:${opencv_COMPILE_OPTIONS_C_RELWITHDEBINFO}>")

set(opencv_LINKER_FLAGS_RELWITHDEBINFO
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${opencv_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${opencv_SHARED_LINK_FLAGS_RELWITHDEBINFO}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${opencv_EXE_LINK_FLAGS_RELWITHDEBINFO}>")

set(opencv_FRAMEWORKS_FOUND_RELWITHDEBINFO "") # Will be filled later
conan_find_apple_frameworks(opencv_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_FRAMEWORK_DIRS_RELWITHDEBINFO}")

# Gather all the libraries that should be linked to the targets (do not touch existing variables)
set(_opencv_DEPENDENCIES_RELWITHDEBINFO "${opencv_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_SYSTEM_LIBS_RELWITHDEBINFO} ZLIB::ZLIB;PNG::PNG;JPEG::JPEG;TIFF::TIFF;libwebp::libwebp;Jasper::Jasper;Jasper::Jasper;ZLIB::ZLIB;ZLIB::ZLIB;PNG::PNG;JPEG::JPEG;TIFF::TIFF;libwebp::libwebp;Jasper::Jasper;Jasper::Jasper;ZLIB::ZLIB;opencv::opencv_highgui;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_imgcodecs;opencv::opencv_videoio;opencv::opencv_video;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_stitching;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_objdetect;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_calib3d;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_videoio;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_imgcodecs;opencv::opencv_imgcodecs;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_photo;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_ml;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_core;opencv::opencv_core;opencv::opencv_highgui;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_imgcodecs;opencv::opencv_videoio;opencv::opencv_video;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_stitching;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_objdetect;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_calib3d;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_videoio;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_imgcodecs;opencv::opencv_imgcodecs;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_photo;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_ml;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_core;opencv::opencv_core;ZLIB::ZLIB;opencv::opencv_core;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_core;opencv::opencv_ml;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_photo;Jasper::Jasper;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;ZLIB::ZLIB;PNG::PNG;JPEG::JPEG;TIFF::TIFF;libwebp::libwebp;Jasper::Jasper;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_imgcodecs;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_imgcodecs;opencv::opencv_videoio;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_objdetect;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_stitching;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_video;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_imgcodecs;opencv::opencv_videoio;opencv::opencv_highgui")

set(opencv_LIBRARIES_TARGETS_RELWITHDEBINFO "") # Will be filled later
set(opencv_LIBRARIES_RELWITHDEBINFO "") # Will be filled later
conan_package_library_targets("${opencv_LIBS_RELWITHDEBINFO}"    # libraries
                              "${opencv_LIB_DIRS_RELWITHDEBINFO}" # package_libdir
                              "${_opencv_DEPENDENCIES_RELWITHDEBINFO}" # deps
                              opencv_LIBRARIES_RELWITHDEBINFO   # out_libraries
                              opencv_LIBRARIES_TARGETS_RELWITHDEBINFO  # out_libraries_targets
                              "_RELWITHDEBINFO"  # config_suffix
                              "opencv")    # package_name

foreach(_FRAMEWORK ${opencv_FRAMEWORKS_FOUND_RELWITHDEBINFO})
    list(APPEND opencv_LIBRARIES_TARGETS_RELWITHDEBINFO ${_FRAMEWORK})
    list(APPEND opencv_LIBRARIES_RELWITHDEBINFO ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${opencv_SYSTEM_LIBS_RELWITHDEBINFO})
    list(APPEND opencv_LIBRARIES_TARGETS_RELWITHDEBINFO ${_SYSTEM_LIB})
    list(APPEND opencv_LIBRARIES_RELWITHDEBINFO ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(opencv_LIBRARIES_TARGETS_RELWITHDEBINFO "${opencv_LIBRARIES_TARGETS_RELWITHDEBINFO};ZLIB::ZLIB;PNG::PNG;JPEG::JPEG;TIFF::TIFF;libwebp::libwebp;Jasper::Jasper;Jasper::Jasper;ZLIB::ZLIB;ZLIB::ZLIB;PNG::PNG;JPEG::JPEG;TIFF::TIFF;libwebp::libwebp;Jasper::Jasper;Jasper::Jasper;ZLIB::ZLIB;opencv::opencv_highgui;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_imgcodecs;opencv::opencv_videoio;opencv::opencv_video;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_stitching;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_objdetect;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_calib3d;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_videoio;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_imgcodecs;opencv::opencv_imgcodecs;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_photo;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_ml;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_core;opencv::opencv_core;opencv::opencv_highgui;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_imgcodecs;opencv::opencv_videoio;opencv::opencv_video;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_stitching;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_objdetect;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_calib3d;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_videoio;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_imgcodecs;opencv::opencv_imgcodecs;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_photo;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_ml;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_core;opencv::opencv_core;ZLIB::ZLIB;opencv::opencv_core;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_core;opencv::opencv_ml;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_photo;Jasper::Jasper;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;ZLIB::ZLIB;PNG::PNG;JPEG::JPEG;TIFF::TIFF;libwebp::libwebp;Jasper::Jasper;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_imgcodecs;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_imgcodecs;opencv::opencv_videoio;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_objdetect;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_stitching;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_video;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_imgcodecs;opencv::opencv_videoio;opencv::opencv_highgui")
set(opencv_LIBRARIES_RELWITHDEBINFO "${opencv_LIBRARIES_RELWITHDEBINFO};ZLIB::ZLIB;PNG::PNG;JPEG::JPEG;TIFF::TIFF;libwebp::libwebp;Jasper::Jasper;Jasper::Jasper;ZLIB::ZLIB;ZLIB::ZLIB;PNG::PNG;JPEG::JPEG;TIFF::TIFF;libwebp::libwebp;Jasper::Jasper;Jasper::Jasper;ZLIB::ZLIB;opencv::opencv_highgui;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_imgcodecs;opencv::opencv_videoio;opencv::opencv_video;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_stitching;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_objdetect;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_calib3d;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_videoio;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_imgcodecs;opencv::opencv_imgcodecs;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_photo;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_ml;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_core;opencv::opencv_core;opencv::opencv_highgui;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_imgcodecs;opencv::opencv_videoio;opencv::opencv_video;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_stitching;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_objdetect;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_calib3d;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_videoio;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_imgcodecs;opencv::opencv_imgcodecs;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_photo;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_ml;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_core;opencv::opencv_core;ZLIB::ZLIB;opencv::opencv_core;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_core;opencv::opencv_ml;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_photo;Jasper::Jasper;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;ZLIB::ZLIB;PNG::PNG;JPEG::JPEG;TIFF::TIFF;libwebp::libwebp;Jasper::Jasper;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_imgcodecs;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_imgcodecs;opencv::opencv_videoio;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_objdetect;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_stitching;opencv::opencv_core;opencv::opencv_flann;opencv::opencv_imgproc;opencv::opencv_features2d;opencv::opencv_calib3d;opencv::opencv_video;opencv::opencv_core;opencv::opencv_imgproc;opencv::opencv_imgcodecs;opencv::opencv_videoio;opencv::opencv_highgui")

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${opencv_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH ${opencv_BUILD_DIRS_RELWITHDEBINFO} ${CMAKE_PREFIX_PATH})
########## COMPONENT opencv::opencv_highgui_alias FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_highgui_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_highgui_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_highgui_alias_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_highgui_alias_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_highgui_alias_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_highgui_alias_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_highgui_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_highgui_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_highgui_alias_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_highgui_alias_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_highgui_alias_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_highgui_alias_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_highgui_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_highgui_alias_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_highgui_alias_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_highgui_alias")

set(opencv_opencv_opencv_highgui_alias_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_highgui_alias_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_highgui_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_highgui FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_highgui_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_highgui_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_highgui_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_highgui_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_highgui_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_highgui_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_highgui_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_highgui_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_highgui_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_highgui_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_highgui_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_highgui_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_highgui_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_highgui_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_highgui_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_highgui")

set(opencv_opencv_opencv_highgui_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_highgui_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_highgui_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_video_alias FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_video_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_video_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_video_alias_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_video_alias_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_video_alias_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_video_alias_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_video_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_video_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_video_alias_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_video_alias_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_video_alias_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_video_alias_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_video_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_video_alias_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_video_alias_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_video_alias")

set(opencv_opencv_opencv_video_alias_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_video_alias_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_video_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_video FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_video_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_video_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_video_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_video_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_video_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_video_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_video_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_video_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_video_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_video_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_video_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_video_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_video_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_video_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_video_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_video")

set(opencv_opencv_opencv_video_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_video_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_video_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_stitching_alias FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_stitching_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_stitching_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_stitching_alias_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_stitching_alias_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_stitching_alias_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_stitching_alias_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_stitching_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_stitching_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_stitching_alias_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_stitching_alias_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_stitching_alias_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_stitching_alias_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_stitching_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_stitching_alias_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_stitching_alias_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_stitching_alias")

set(opencv_opencv_opencv_stitching_alias_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_stitching_alias_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_stitching_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_stitching FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_stitching_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_stitching_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_stitching_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_stitching_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_stitching_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_stitching_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_stitching_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_stitching_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_stitching_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_stitching_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_stitching_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_stitching_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_stitching_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_stitching_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_stitching_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_stitching")

set(opencv_opencv_opencv_stitching_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_stitching_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_stitching_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_objdetect_alias FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_objdetect_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_objdetect_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_objdetect_alias_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_objdetect_alias_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_objdetect_alias_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_objdetect_alias_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_objdetect_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_objdetect_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_objdetect_alias_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_objdetect_alias_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_objdetect_alias_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_objdetect_alias_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_objdetect_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_objdetect_alias_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_objdetect_alias_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_objdetect_alias")

set(opencv_opencv_opencv_objdetect_alias_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_objdetect_alias_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_objdetect_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_objdetect FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_objdetect_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_objdetect_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_objdetect_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_objdetect_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_objdetect_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_objdetect_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_objdetect_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_objdetect_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_objdetect_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_objdetect_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_objdetect_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_objdetect_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_objdetect_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_objdetect_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_objdetect_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_objdetect")

set(opencv_opencv_opencv_objdetect_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_objdetect_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_objdetect_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_calib3d_alias FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_calib3d_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_calib3d_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_calib3d_alias_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_calib3d_alias_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_calib3d_alias_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_calib3d_alias_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_calib3d_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_calib3d_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_calib3d_alias_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_calib3d_alias_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_calib3d_alias_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_calib3d_alias_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_calib3d_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_calib3d_alias_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_calib3d_alias_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_calib3d_alias")

set(opencv_opencv_opencv_calib3d_alias_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_calib3d_alias_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_calib3d_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_calib3d FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_calib3d_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_calib3d_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_calib3d_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_calib3d_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_calib3d_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_calib3d_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_calib3d_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_calib3d_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_calib3d_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_calib3d_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_calib3d_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_calib3d_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_calib3d_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_calib3d_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_calib3d_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_calib3d")

set(opencv_opencv_opencv_calib3d_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_calib3d_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_calib3d_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_videoio_alias FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_videoio_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_videoio_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_videoio_alias_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_videoio_alias_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_videoio_alias_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_videoio_alias_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_videoio_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_videoio_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_videoio_alias_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_videoio_alias_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_videoio_alias_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_videoio_alias_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_videoio_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_videoio_alias_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_videoio_alias_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_videoio_alias")

set(opencv_opencv_opencv_videoio_alias_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_videoio_alias_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_videoio_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_videoio FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_videoio_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_videoio_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_videoio_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_videoio_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_videoio_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_videoio_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_videoio_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_videoio_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_videoio_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_videoio_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_videoio_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_videoio_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_videoio_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_videoio_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_videoio_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_videoio")

set(opencv_opencv_opencv_videoio_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_videoio_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_videoio_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_imgcodecs_alias FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_imgcodecs_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_imgcodecs_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_imgcodecs_alias_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_imgcodecs_alias_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_imgcodecs_alias_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_imgcodecs_alias_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_imgcodecs_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_imgcodecs_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_imgcodecs_alias_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_imgcodecs_alias_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_imgcodecs_alias_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_imgcodecs_alias_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_imgcodecs_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_imgcodecs_alias_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_imgcodecs_alias_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_imgcodecs_alias")

set(opencv_opencv_opencv_imgcodecs_alias_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_imgcodecs_alias_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_imgcodecs_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_imgcodecs FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_imgcodecs_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_imgcodecs_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_imgcodecs_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_imgcodecs_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_imgcodecs_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_imgcodecs_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_imgcodecs_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_imgcodecs_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_imgcodecs_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_imgcodecs_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_imgcodecs_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_imgcodecs_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_imgcodecs_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_imgcodecs_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_imgcodecs_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_imgcodecs")

set(opencv_opencv_opencv_imgcodecs_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_imgcodecs_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_imgcodecs_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_features2d_alias FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_features2d_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_features2d_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_features2d_alias_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_features2d_alias_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_features2d_alias_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_features2d_alias_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_features2d_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_features2d_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_features2d_alias_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_features2d_alias_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_features2d_alias_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_features2d_alias_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_features2d_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_features2d_alias_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_features2d_alias_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_features2d_alias")

set(opencv_opencv_opencv_features2d_alias_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_features2d_alias_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_features2d_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_features2d FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_features2d_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_features2d_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_features2d_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_features2d_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_features2d_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_features2d_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_features2d_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_features2d_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_features2d_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_features2d_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_features2d_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_features2d_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_features2d_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_features2d_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_features2d_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_features2d")

set(opencv_opencv_opencv_features2d_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_features2d_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_features2d_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_photo_alias FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_photo_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_photo_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_photo_alias_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_photo_alias_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_photo_alias_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_photo_alias_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_photo_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_photo_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_photo_alias_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_photo_alias_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_photo_alias_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_photo_alias_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_photo_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_photo_alias_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_photo_alias_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_photo_alias")

set(opencv_opencv_opencv_photo_alias_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_photo_alias_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_photo_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_photo FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_photo_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_photo_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_photo_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_photo_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_photo_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_photo_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_photo_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_photo_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_photo_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_photo_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_photo_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_photo_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_photo_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_photo_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_photo_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_photo")

set(opencv_opencv_opencv_photo_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_photo_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_photo_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_ml_alias FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_ml_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_ml_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_ml_alias_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_ml_alias_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_ml_alias_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_ml_alias_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_ml_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_ml_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_ml_alias_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_ml_alias_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_ml_alias_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_ml_alias_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_ml_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_ml_alias_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_ml_alias_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_ml_alias")

set(opencv_opencv_opencv_ml_alias_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_ml_alias_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_ml_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_ml FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_ml_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_ml_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_ml_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_ml_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_ml_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_ml_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_ml_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_ml_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_ml_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_ml_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_ml_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_ml_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_ml_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_ml_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_ml_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_ml")

set(opencv_opencv_opencv_ml_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_ml_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_ml_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_imgproc_alias FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_imgproc_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_imgproc_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_imgproc_alias_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_imgproc_alias_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_imgproc_alias_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_imgproc_alias_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_imgproc_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_imgproc_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_imgproc_alias_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_imgproc_alias_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_imgproc_alias_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_imgproc_alias_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_imgproc_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_imgproc_alias_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_imgproc_alias_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_imgproc_alias")

set(opencv_opencv_opencv_imgproc_alias_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_imgproc_alias_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_imgproc_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_imgproc FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_imgproc_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_imgproc_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_imgproc_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_imgproc_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_imgproc_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_imgproc_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_imgproc_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_imgproc_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_imgproc_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_imgproc_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_imgproc_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_imgproc_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_imgproc_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_imgproc_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_imgproc_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_imgproc")

set(opencv_opencv_opencv_imgproc_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_imgproc_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_imgproc_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_flann_alias FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_flann_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_flann_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_flann_alias_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_flann_alias_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_flann_alias_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_flann_alias_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_flann_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_flann_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_flann_alias_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_flann_alias_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_flann_alias_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_flann_alias_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_flann_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_flann_alias_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_flann_alias_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_flann_alias")

set(opencv_opencv_opencv_flann_alias_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_flann_alias_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_flann_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_flann FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_flann_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_flann_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_flann_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_flann_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_flann_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_flann_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_flann_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_flann_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_flann_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_flann_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_flann_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_flann_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_flann_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_flann_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_flann_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_flann")

set(opencv_opencv_opencv_flann_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_flann_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_flann_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_core_alias FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_core_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_core_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_core_alias_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_core_alias_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_core_alias_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_core_alias_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_core_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_core_alias_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_core_alias_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_core_alias_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_core_alias_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_core_alias_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_core_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_core_alias_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_core_alias_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_core_alias")

set(opencv_opencv_opencv_core_alias_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_core_alias_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_core_alias_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})
########## COMPONENT opencv::opencv_core FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(opencv_opencv_opencv_core_FRAMEWORKS_FOUND_RELWITHDEBINFO "")
conan_find_apple_frameworks(opencv_opencv_opencv_core_FRAMEWORKS_FOUND_RELWITHDEBINFO "${opencv_opencv_opencv_core_FRAMEWORKS_RELWITHDEBINFO}" "${opencv_opencv_opencv_core_FRAMEWORK_DIRS_RELWITHDEBINFO}")

set(opencv_opencv_opencv_core_LIB_TARGETS_RELWITHDEBINFO "")
set(opencv_opencv_opencv_core_NOT_USED_RELWITHDEBINFO "")
set(opencv_opencv_opencv_core_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO ${opencv_opencv_opencv_core_FRAMEWORKS_FOUND_RELWITHDEBINFO} ${opencv_opencv_opencv_core_SYSTEM_LIBS_RELWITHDEBINFO} ${opencv_opencv_opencv_core_DEPENDENCIES_RELWITHDEBINFO})
conan_package_library_targets("${opencv_opencv_opencv_core_LIBS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_core_LIB_DIRS_RELWITHDEBINFO}"
                              "${opencv_opencv_opencv_core_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO}"
                              opencv_opencv_opencv_core_NOT_USED_RELWITHDEBINFO
                              opencv_opencv_opencv_core_LIB_TARGETS_RELWITHDEBINFO
                              "_RELWITHDEBINFO"
                              "opencv_opencv_opencv_core")

set(opencv_opencv_opencv_core_LINK_LIBS_RELWITHDEBINFO ${opencv_opencv_opencv_core_LIB_TARGETS_RELWITHDEBINFO} ${opencv_opencv_opencv_core_LIBS_FRAMEWORKS_DEPS_RELWITHDEBINFO})


########## GLOBAL TARGET PROPERTIES RelWithDebInfo ########################################
set_property(TARGET opencv::opencv
             PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_LIBRARIES_TARGETS_RELWITHDEBINFO}
                                           ${opencv_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv
             PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv
             PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv
             PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv
             PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_COMPILE_OPTIONS_RELWITHDEBINFO}> APPEND)

########## COMPONENTS TARGET PROPERTIES RelWithDebInfo ########################################
########## COMPONENT opencv::opencv_highgui_alias TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_highgui_alias PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_highgui_alias_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_highgui_alias_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_highgui_alias PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_highgui_alias_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_highgui_alias PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_highgui_alias_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_highgui_alias PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_highgui_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_highgui_alias PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_highgui_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_highgui_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_highgui_alias_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_highgui TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_highgui PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_highgui_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_highgui_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_highgui PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_highgui_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_highgui PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_highgui_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_highgui PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_highgui_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_highgui PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_highgui_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_highgui_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_highgui_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_video_alias TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_video_alias PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_video_alias_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_video_alias_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_video_alias PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_video_alias_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_video_alias PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_video_alias_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_video_alias PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_video_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_video_alias PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_video_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_video_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_video_alias_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_video TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_video PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_video_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_video_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_video PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_video_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_video PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_video_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_video PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_video_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_video PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_video_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_video_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_video_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_stitching_alias TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_stitching_alias PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_stitching_alias_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_stitching_alias_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_stitching_alias PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_stitching_alias_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_stitching_alias PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_stitching_alias_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_stitching_alias PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_stitching_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_stitching_alias PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_stitching_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_stitching_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_stitching_alias_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_stitching TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_stitching PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_stitching_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_stitching_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_stitching PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_stitching_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_stitching PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_stitching_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_stitching PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_stitching_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_stitching PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_stitching_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_stitching_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_stitching_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_objdetect_alias TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_objdetect_alias PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_objdetect_alias_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_objdetect_alias_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_objdetect_alias PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_objdetect_alias_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_objdetect_alias PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_objdetect_alias_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_objdetect_alias PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_objdetect_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_objdetect_alias PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_objdetect_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_objdetect_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_objdetect_alias_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_objdetect TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_objdetect PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_objdetect_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_objdetect_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_objdetect PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_objdetect_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_objdetect PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_objdetect_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_objdetect PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_objdetect_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_objdetect PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_objdetect_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_objdetect_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_objdetect_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_calib3d_alias TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_calib3d_alias PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_calib3d_alias_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_calib3d_alias_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_calib3d_alias PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_calib3d_alias_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_calib3d_alias PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_calib3d_alias_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_calib3d_alias PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_calib3d_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_calib3d_alias PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_calib3d_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_calib3d_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_calib3d_alias_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_calib3d TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_calib3d PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_calib3d_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_calib3d_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_calib3d PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_calib3d_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_calib3d PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_calib3d_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_calib3d PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_calib3d_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_calib3d PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_calib3d_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_calib3d_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_calib3d_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_videoio_alias TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_videoio_alias PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_videoio_alias_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_videoio_alias_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_videoio_alias PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_videoio_alias_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_videoio_alias PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_videoio_alias_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_videoio_alias PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_videoio_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_videoio_alias PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_videoio_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_videoio_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_videoio_alias_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_videoio TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_videoio PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_videoio_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_videoio_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_videoio PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_videoio_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_videoio PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_videoio_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_videoio PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_videoio_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_videoio PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_videoio_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_videoio_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_videoio_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_imgcodecs_alias TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_imgcodecs_alias PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_imgcodecs_alias_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_imgcodecs_alias_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_imgcodecs_alias PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_imgcodecs_alias_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_imgcodecs_alias PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_imgcodecs_alias_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_imgcodecs_alias PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_imgcodecs_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_imgcodecs_alias PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_imgcodecs_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_imgcodecs_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_imgcodecs_alias_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_imgcodecs TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_imgcodecs PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_imgcodecs_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_imgcodecs_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_imgcodecs PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_imgcodecs_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_imgcodecs PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_imgcodecs_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_imgcodecs PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_imgcodecs_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_imgcodecs PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_imgcodecs_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_imgcodecs_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_imgcodecs_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_features2d_alias TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_features2d_alias PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_features2d_alias_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_features2d_alias_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_features2d_alias PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_features2d_alias_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_features2d_alias PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_features2d_alias_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_features2d_alias PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_features2d_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_features2d_alias PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_features2d_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_features2d_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_features2d_alias_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_features2d TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_features2d PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_features2d_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_features2d_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_features2d PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_features2d_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_features2d PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_features2d_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_features2d PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_features2d_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_features2d PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_features2d_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_features2d_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_features2d_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_photo_alias TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_photo_alias PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_photo_alias_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_photo_alias_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_photo_alias PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_photo_alias_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_photo_alias PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_photo_alias_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_photo_alias PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_photo_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_photo_alias PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_photo_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_photo_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_photo_alias_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_photo TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_photo PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_photo_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_photo_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_photo PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_photo_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_photo PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_photo_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_photo PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_photo_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_photo PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_photo_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_photo_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_photo_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_ml_alias TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_ml_alias PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_ml_alias_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_ml_alias_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_ml_alias PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_ml_alias_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_ml_alias PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_ml_alias_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_ml_alias PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_ml_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_ml_alias PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_ml_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_ml_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_ml_alias_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_ml TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_ml PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_ml_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_ml_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_ml PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_ml_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_ml PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_ml_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_ml PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_ml_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_ml PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_ml_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_ml_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_ml_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_imgproc_alias TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_imgproc_alias PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_imgproc_alias_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_imgproc_alias_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_imgproc_alias PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_imgproc_alias_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_imgproc_alias PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_imgproc_alias_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_imgproc_alias PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_imgproc_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_imgproc_alias PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_imgproc_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_imgproc_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_imgproc_alias_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_imgproc TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_imgproc PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_imgproc_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_imgproc_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_imgproc PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_imgproc_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_imgproc PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_imgproc_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_imgproc PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_imgproc_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_imgproc PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_imgproc_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_imgproc_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_imgproc_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_flann_alias TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_flann_alias PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_flann_alias_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_flann_alias_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_flann_alias PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_flann_alias_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_flann_alias PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_flann_alias_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_flann_alias PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_flann_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_flann_alias PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_flann_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_flann_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_flann_alias_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_flann TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_flann PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_flann_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_flann_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_flann PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_flann_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_flann PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_flann_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_flann PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_flann_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_flann PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_flann_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_flann_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_flann_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_core_alias TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_core_alias PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_core_alias_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_core_alias_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_core_alias PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_core_alias_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_core_alias PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_core_alias_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_core_alias PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_core_alias_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_core_alias PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_core_alias_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_core_alias_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_core_alias_TARGET_PROPERTIES TRUE)
########## COMPONENT opencv::opencv_core TARGET PROPERTIES ######################################
set_property(TARGET opencv::opencv_core PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_core_LINK_LIBS_RELWITHDEBINFO}
             ${opencv_opencv_opencv_core_OBJECTS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_core PROPERTY INTERFACE_LINK_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_core_LINKER_FLAGS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_core PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_core_INCLUDE_DIRS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_core PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:RelWithDebInfo>:${opencv_opencv_opencv_core_COMPILE_DEFINITIONS_RELWITHDEBINFO}> APPEND)
set_property(TARGET opencv::opencv_core PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:RelWithDebInfo>:
             ${opencv_opencv_opencv_core_COMPILE_OPTIONS_C_RELWITHDEBINFO}
             ${opencv_opencv_opencv_core_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}> APPEND)
set(opencv_opencv_opencv_core_TARGET_PROPERTIES TRUE)