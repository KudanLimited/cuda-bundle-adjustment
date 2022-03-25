cmake_minimum_required(VERSION 3.16)

project(sample_ba_from_file LANGUAGES CXX CUDA)

set(SAMPLE_UTILITY_DIR ../utility)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
endif()

if (WIN32)
	add_compile_options(/wd4819)
	add_definitions(-D_SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING)
endif()

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "Auto")

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${SAMPLE_UTILITY_DIR})

file(GLOB srcs ./*.cpp ./*.h* ${SAMPLE_UTILITY_DIR}/*.cpp ${SAMPLE_UTILITY_DIR}/*.h*)
add_executable(sample_ba_from_file ${srcs})
target_compile_features(sample_ba_from_file PUBLIC cxx_std_11)
target_link_libraries(sample_ba_from_file opencv::opencv cuda_bundle_adjustment Eigen3::Eigen3)


