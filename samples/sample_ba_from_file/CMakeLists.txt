cmake_minimum_required(VERSION 3.16)

project(sample_ba_from_file LANGUAGES CXX)

set(SAMPLE_UTILITY_DIR ../utility)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
endif()

if (WIN32)
	add_definitions(-D_SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING)
endif()

# CUDA
find_package(CUDAToolkit REQUIRED)

file(GLOB srcs ./*.cpp ./*.h* ${SAMPLE_UTILITY_DIR}/*.cpp ${SAMPLE_UTILITY_DIR}/*.h*)

add_executable(sample_ba_from_file ${srcs})

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

target_include_directories(
	sample_ba_from_file 
	PUBLIC 
	"${CMAKE_SOURCE_DIR}/include" 
	${SAMPLE_UTILITY_DIR}
	${CUDAToolkit_INCLUDE_DIRS}
	${Eigen3_INCLUDE_DIR}
	${opencv_INCLUDE_DIRS}
)
target_compile_features(sample_ba_from_file PUBLIC cxx_std_11)
target_link_libraries(
	sample_ba_from_file 
	cugo
	opencv::opencv 
	Eigen3::Eigen3
)

