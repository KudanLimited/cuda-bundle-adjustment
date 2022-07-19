cmake_minimum_required(VERSION 3.16)

project(CudaGraphOptimisation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_FLOAT32 "Use 32bit float in internal floating-point operations (default is 64bit float)" OFF)
option(USE_TOOLKIT_10_2 "Enable CUDA Toolkit 10.2 compatibility." OFF)
option(BUILD_CUGO_SHARED "Enable to building of cugo as a shared library" OFF)
option(USE_ZERO_COPY "Enables zero copy for devices with integrated GPUs" OFF)

add_subdirectory(src)

if(ENABLE_SAMPLES)
	message("Configuring samples...")
	add_subdirectory(samples/sample_ba_from_file)
	if(WITH_G2O)
		add_subdirectory(samples/sample_comparison_with_g2o)
	endif()
endif()

unset(USE_FLOAT32 CACHE)
unset(USE_TOOLKIT_10_2 CACHE)
unset(BUILD_CUGO_SHARED CACHE)
unset(USE_ZERO_COPY CACHE)
