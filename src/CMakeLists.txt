cmake_minimum_required(VERSION 3.16)

project(cuda_bundle_adjustment LANGUAGES CXX CUDA)

set(CUBA_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
include_directories(${CUBA_INCLUDE_DIR})

#CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "Auto")
set(CUDA_ARCH "${ARCH_FLAGS}" CACHE STRING "Value of the NVCC -arch option." FORCE)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${ARCH_FLAGS} -lineinfo --diag_suppress=esa_on_defaulted_function_ignored")
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G -lineinfo")

message(STATUS "CUDA_ARCH: \"${CUDA_ARCH}\"")

# Flags for each platform
if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
endif()

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

if (WIN32)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -dc -Xcompiler \"/wd 4819\" -Xcompiler \"/wd 4244\"")
	add_compile_options(/wd4819)
	add_definitions(-D_SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING)
endif()

if (USE_FLOAT32)
	add_definitions(-DUSE_FLOAT32)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

file(GLOB_RECURSE srcs ./*.cpp ./*.cu ./*.h ${CUBA_INCLUDE_DIR}/*.h)
add_library(cuda_bundle_adjustment STATIC ${srcs})

target_compile_features(cuda_bundle_adjustment PUBLIC cxx_std_11)
target_link_libraries(cuda_bundle_adjustment PUBLIC ${CUDA_cusparse_LIBRARY} ${CUDA_cusolver_LIBRARY} Eigen3::Eigen3)
target_include_directories(cuda_bundle_adjustment PUBLIC "${CMAKE_SOURCE_DIR}/src")
#set_target_properties(cuda_bundle_adjustment PROPERTIES CUDA_SEPARABLE_COMPILATION ON) 
#set_target_properties(cuda_bundle_adjustment PROPERTIES POSITION_INDEPENDENT_CODE ON)
#set_target_properties(cuda_bundle_adjustment PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

install(
	TARGETS cuda_bundle_adjustment
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

install(
	DIRECTORY ${CUBA_INCLUDE_DIR}
	DESTINATION ${CMAKE_INSTALL_PREFIX}
	FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
