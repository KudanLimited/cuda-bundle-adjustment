

# Conan automatically generated toolchain file
# DO NOT EDIT MANUALLY, it will be overwritten

# Avoid including toolchain file several times (bad if appending to variables like
#   CMAKE_CXX_FLAGS. See https://github.com/android/ndk/issues/323
include_guard()

if(CMAKE_TOOLCHAIN_FILE)
    message("Using Conan toolchain: ${CMAKE_TOOLCHAIN_FILE}.")
endif()





set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)


message(STATUS "Conan toolchain: Setting CMAKE_POSITION_INDEPENDENT_CODE=ON (options.fPIC)")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


set(CONAN_CXX_FLAGS "${CONAN_CXX_FLAGS} -m64")
set(CONAN_C_FLAGS "${CONAN_C_FLAGS} -m64")
set(CONAN_SHARED_LINKER_FLAGS "${CONAN_SHARED_LINKER_FLAGS} -m64")
set(CONAN_EXE_LINKER_FLAGS "${CONAN_EXE_LINKER_FLAGS} -m64")

set(CONAN_CXX_FLAGS "${CONAN_CXX_FLAGS} -stdlib=libstdc++")


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
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR} "/home/kudan/.conan/data/opencv/4.0.1/_/_/package/4372ad2209849231b1f53102f661fd3c9f035e6d/lib/cmake" "/home/kudan/.conan/data/zlib/1.2.11/_/_/package/d988447fa516eac7400b2f34e2d4b89e42b4b1a8/" "/home/kudan/.conan/data/libjpeg/9d/_/_/package/d988447fa516eac7400b2f34e2d4b89e42b4b1a8/" "/home/kudan/.conan/data/libtiff/4.1.0/_/_/package/ca7f9ce2c8d2270a71778d92f60dcceffe9cfc54/" "/home/kudan/.conan/data/libwebp/1.2.0/_/_/package/f67d7d7df8619b72b467fed73316fd5a7411cd95/" "/home/kudan/.conan/data/libpng/1.6.37/_/_/package/4305977d9227c3156a3f9d8759c25afec76d9f0a/" "/home/kudan/.conan/data/jasper/2.0.32/_/_/package/a7d44ee787bf104410d5584aaf0c6874ece2846e/" "/home/kudan/.conan/data/jasper/2.0.32/_/_/package/a7d44ee787bf104410d5584aaf0c6874ece2846e/lib/cmake" "/home/kudan/.conan/data/xz_utils/5.2.5/_/_/package/d988447fa516eac7400b2f34e2d4b89e42b4b1a8/" "/home/kudan/.conan/data/xz_utils/5.2.5/_/_/package/d988447fa516eac7400b2f34e2d4b89e42b4b1a8/lib/cmake" "/home/kudan/.conan/data/jbig/20160605/_/_/package/ef361fa1f73ddde93e03f3b5d0a6a285a0bb9aa9/" "/home/kudan/.conan/data/zstd/1.5.1/_/_/package/d988447fa516eac7400b2f34e2d4b89e42b4b1a8/" ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_LIST_DIR} "/home/kudan/.conan/data/opencv/4.0.1/_/_/package/4372ad2209849231b1f53102f661fd3c9f035e6d/lib/cmake" "/home/kudan/.conan/data/zlib/1.2.11/_/_/package/d988447fa516eac7400b2f34e2d4b89e42b4b1a8/" "/home/kudan/.conan/data/libjpeg/9d/_/_/package/d988447fa516eac7400b2f34e2d4b89e42b4b1a8/" "/home/kudan/.conan/data/libtiff/4.1.0/_/_/package/ca7f9ce2c8d2270a71778d92f60dcceffe9cfc54/" "/home/kudan/.conan/data/libwebp/1.2.0/_/_/package/f67d7d7df8619b72b467fed73316fd5a7411cd95/" "/home/kudan/.conan/data/libpng/1.6.37/_/_/package/4305977d9227c3156a3f9d8759c25afec76d9f0a/" "/home/kudan/.conan/data/jasper/2.0.32/_/_/package/a7d44ee787bf104410d5584aaf0c6874ece2846e/" "/home/kudan/.conan/data/jasper/2.0.32/_/_/package/a7d44ee787bf104410d5584aaf0c6874ece2846e/lib/cmake" "/home/kudan/.conan/data/xz_utils/5.2.5/_/_/package/d988447fa516eac7400b2f34e2d4b89e42b4b1a8/" "/home/kudan/.conan/data/xz_utils/5.2.5/_/_/package/d988447fa516eac7400b2f34e2d4b89e42b4b1a8/lib/cmake" "/home/kudan/.conan/data/jbig/20160605/_/_/package/ef361fa1f73ddde93e03f3b5d0a6a285a0bb9aa9/" "/home/kudan/.conan/data/zstd/1.5.1/_/_/package/d988447fa516eac7400b2f34e2d4b89e42b4b1a8/" ${CMAKE_PREFIX_PATH})

# To support cross building to iOS, watchOS and tvOS where CMake looks for config files
# only in the system frameworks unless you declare the XXX_DIR variables




message(STATUS "Conan toolchain: Setting BUILD_SHARED_LIBS = OFF")
set(BUILD_SHARED_LIBS OFF)

# Variables
set(ENABLE_SAMPLES "False" CACHE STRING "Variable ENABLE_SAMPLES conan-toolchain defined")
set(WITH_G2O "False" CACHE STRING "Variable WITH_G2O conan-toolchain defined")
set(USE_FLOAT32 "False" CACHE STRING "Variable USE_FLOAT32 conan-toolchain defined")
# Variables  per configuration


# Preprocessor definitions
# Preprocessor definitions per configuration
