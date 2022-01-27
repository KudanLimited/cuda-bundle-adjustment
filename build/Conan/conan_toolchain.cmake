

# Conan automatically generated toolchain file
# DO NOT EDIT MANUALLY, it will be overwritten

# Avoid including toolchain file several times (bad if appending to variables like
#   CMAKE_CXX_FLAGS. See https://github.com/android/ndk/issues/323
include_guard()

if(CMAKE_TOOLCHAIN_FILE)
    message("Using Conan toolchain: ${CMAKE_TOOLCHAIN_FILE}.")
endif()





set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)


message(STATUS "Conan toolchain: Setting CMAKE_POSITION_INDEPENDENT_CODE=ON (options.fPIC)")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


set(CONAN_CXX_FLAGS "${CONAN_CXX_FLAGS} -m64")
set(CONAN_C_FLAGS "${CONAN_C_FLAGS} -m64")
set(CONAN_SHARED_LINKER_FLAGS "${CONAN_SHARED_LINKER_FLAGS} -m64")
set(CONAN_EXE_LINKER_FLAGS "${CONAN_EXE_LINKER_FLAGS} -m64")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)


set(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} ${CONAN_CXX_FLAGS}")
set(CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT} ${CONAN_C_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${CMAKE_SHARED_LINKER_FLAGS_INIT} ${CONAN_SHARED_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_INIT "${CMAKE_EXE_LINKER_FLAGS_INIT} ${CONAN_EXE_LINKER_FLAGS}")

get_property( _CMAKE_IN_TRY_COMPILE GLOBAL PROPERTY IN_TRY_COMPILE )
if(_CMAKE_IN_TRY_COMPILE)
    message(STATUS "Running toolchain IN_TRY_COMPILE")
    return()
endif()

set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)
# To support the generators based on find_package()
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR} "/home/kudan/.conan/data/opencv/4.0.1/_/_/package/8c33e7eafe6a9e7de92282766132db65ff6b6cc9/lib/cmake" "/home/kudan/.conan/data/g2o/20201223/_/_/package/5e481ef36063949e2b86820fabbf7b010b748072/" "/home/kudan/.conan/data/zlib/1.2.11/_/_/package/6b7ff26bfd4c2cf2ccba522bfba2d2e7820e40da/" "/home/kudan/.conan/data/libjpeg/9d/_/_/package/6b7ff26bfd4c2cf2ccba522bfba2d2e7820e40da/" "/home/kudan/.conan/data/libtiff/4.3.0/_/_/package/9bdae2649e516367a6bbbaaf5857a150ae54c8d3/" "/home/kudan/.conan/data/libwebp/1.2.0/_/_/package/701d9c6d6d4798bc8d444813c228ba745babc496/" "/home/kudan/.conan/data/libpng/1.6.37/_/_/package/1e8b7ff23bd5e7932ba7e8874349125fdf8e91ec/" "/home/kudan/.conan/data/jasper/2.0.32/_/_/package/cb76f1272041e14f4d9aa410a2b335537c1d5bde/" "/home/kudan/.conan/data/jasper/2.0.32/_/_/package/cb76f1272041e14f4d9aa410a2b335537c1d5bde/lib/cmake" "/home/kudan/.conan/data/libdeflate/1.8/_/_/package/6b7ff26bfd4c2cf2ccba522bfba2d2e7820e40da/" "/home/kudan/.conan/data/xz_utils/5.2.5/_/_/package/6b7ff26bfd4c2cf2ccba522bfba2d2e7820e40da/" "/home/kudan/.conan/data/xz_utils/5.2.5/_/_/package/6b7ff26bfd4c2cf2ccba522bfba2d2e7820e40da/lib/cmake" "/home/kudan/.conan/data/jbig/20160605/_/_/package/bc25cccf370acdc9aede786b6cfe7144959ff365/" "/home/kudan/.conan/data/zstd/1.5.1/_/_/package/6b7ff26bfd4c2cf2ccba522bfba2d2e7820e40da/" ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_LIST_DIR} "/home/kudan/.conan/data/opencv/4.0.1/_/_/package/8c33e7eafe6a9e7de92282766132db65ff6b6cc9/lib/cmake" "/home/kudan/.conan/data/g2o/20201223/_/_/package/5e481ef36063949e2b86820fabbf7b010b748072/" "/home/kudan/.conan/data/zlib/1.2.11/_/_/package/6b7ff26bfd4c2cf2ccba522bfba2d2e7820e40da/" "/home/kudan/.conan/data/libjpeg/9d/_/_/package/6b7ff26bfd4c2cf2ccba522bfba2d2e7820e40da/" "/home/kudan/.conan/data/libtiff/4.3.0/_/_/package/9bdae2649e516367a6bbbaaf5857a150ae54c8d3/" "/home/kudan/.conan/data/libwebp/1.2.0/_/_/package/701d9c6d6d4798bc8d444813c228ba745babc496/" "/home/kudan/.conan/data/libpng/1.6.37/_/_/package/1e8b7ff23bd5e7932ba7e8874349125fdf8e91ec/" "/home/kudan/.conan/data/jasper/2.0.32/_/_/package/cb76f1272041e14f4d9aa410a2b335537c1d5bde/" "/home/kudan/.conan/data/jasper/2.0.32/_/_/package/cb76f1272041e14f4d9aa410a2b335537c1d5bde/lib/cmake" "/home/kudan/.conan/data/libdeflate/1.8/_/_/package/6b7ff26bfd4c2cf2ccba522bfba2d2e7820e40da/" "/home/kudan/.conan/data/xz_utils/5.2.5/_/_/package/6b7ff26bfd4c2cf2ccba522bfba2d2e7820e40da/" "/home/kudan/.conan/data/xz_utils/5.2.5/_/_/package/6b7ff26bfd4c2cf2ccba522bfba2d2e7820e40da/lib/cmake" "/home/kudan/.conan/data/jbig/20160605/_/_/package/bc25cccf370acdc9aede786b6cfe7144959ff365/" "/home/kudan/.conan/data/zstd/1.5.1/_/_/package/6b7ff26bfd4c2cf2ccba522bfba2d2e7820e40da/" ${CMAKE_PREFIX_PATH})

# To support cross building to iOS, watchOS and tvOS where CMake looks for config files
# only in the system frameworks unless you declare the XXX_DIR variables




message(STATUS "Conan toolchain: Setting BUILD_SHARED_LIBS = OFF")
set(BUILD_SHARED_LIBS OFF)

# Variables
set(ENABLE_SAMPLES "False" CACHE STRING "Variable ENABLE_SAMPLES conan-toolchain defined")
set(WITH_G2O "True" CACHE STRING "Variable WITH_G2O conan-toolchain defined")
set(USE_FLOAT32 "False" CACHE STRING "Variable USE_FLOAT32 conan-toolchain defined")
# Variables  per configuration


# Preprocessor definitions
# Preprocessor definitions per configuration
