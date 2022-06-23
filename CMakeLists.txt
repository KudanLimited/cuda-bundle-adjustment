cmake_minimum_required(VERSION 3.16)

project(CudaGraphOptimisation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_FLOAT32 "Use 32bit float in internal floating-point operations (default is 64bit float)" OFF)

add_subdirectory(src)

if(ENABLE_SAMPLES)
	message("Configuring samples...")
	add_subdirectory(samples/sample_ba_from_file)
	if(WITH_G2O)
		add_subdirectory(samples/sample_comparison_with_g2o)
	endif()
endif()
