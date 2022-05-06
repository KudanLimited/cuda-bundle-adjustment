cmake_minimum_required(VERSION 3.16)

project(CudaBundleAdjustment)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ENABLE_SAMPLES FALSE)
set(WITH_G2O FALSE)
option(USE_FLOAT32    "Use 32bit float in internal floating-point operations (default is 64bit float)" OFF)

# Eigen
find_package(Eigen3 CONFIG REQUIRED)
include_directories(${eigen_INCLUDE_DIRS_RELWITHDEBINFO})
message("${eigen_INCLUDE_DIRS_RELWITHDEBINFO}")

add_subdirectory(src)

if(ENABLE_SAMPLES)
	# OpenCV
	find_package(OpenCV REQUIRED)
	include_directories(${OpenCV_INCLUDE_DIRS})

	message("Configuring samples...")
	add_subdirectory(samples/sample_ba_from_file)
	if(WITH_G2O)
		add_subdirectory(samples/sample_comparison_with_g2o)
	endif()
endif()
