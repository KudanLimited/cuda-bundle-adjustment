cmake_minimum_required(VERSION 3.16)

project(cugo LANGUAGES CXX CUDA)

set(CUGO_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

find_package(Eigen3 REQUIRED)

if (DEBUG_CUGO)
	set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G -lineinfo")
else()
	set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-lineinfo")
endif()

set(CUGO_COMPILE_FLAGS "")
set(CUGO_DEFINITIONS "")

if (CMAKE_COMPILER_IS_GNUCXX)
	list(APPEND CUGO_COMPILE_FLAGS "-O3 -Wall")
endif()

find_package(CUDAToolkit REQUIRED)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61 75)
endif()
message(STATUS "CUDA Found: ${CUDAToolkit_FOUND}")
message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")

if (WIN32)
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -dc -Xcompiler \"/wd 4819\" -Xcompiler \"/wd 4244\"")
	list(APPEND CUGO_DEFINITIONS -D_SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING -D_USE_MATH_DEFINES=1)
endif()

if (USE_FLOAT32)
	add_definitions(-DUSE_FLOAT32)
endif()

file(GLOB_RECURSE srcs ./*.cpp ./*.h ./*.cu ${CUGO_INCLUDE_DIR}/*.h)

if (BUILD_CUGO_SHARED) 
	message("Building CUGO as shared library....")
	add_library(cugo SHARED ${srcs})
else()
	message("Building CUGO as static library....")
	add_library(cugo STATIC ${srcs})
endif()

if (BUILD_CUGO_SHARED OR WIN32)
	set(CUDA_LIBS 
		CUDA::cusparse 
		CUDA::cusolver
		CUDA::cublas
		CUDA::cublasLt 
		CUDA::cudart
	)
else()
	set(CUDA_LIBS 
		CUDA::cusparse_static 
		CUDA::cusolver_static 
		CUDA::cublas_static 
		CUDA::cublasLt_static 
		CUDA::cudart_static
		${CUDAToolkit_LIBRARY_DIR}/libmetis_static.a
	)
endif()

target_include_directories(
	cugo 
	PRIVATE 
	${CUGO_INCLUDE_DIR}
	${Eigen3_INCLUDE_DIR}
)
target_compile_features(
	cugo 
	PUBLIC 
	cxx_std_11
)
target_link_libraries(
	cugo
	PUBLIC 
	${CUDA_LIBS}
	Eigen3::Eigen3
)
target_compile_options(
	cugo 
	PRIVATE 
	${CUGO_COMPILE_FLAGS}
)
target_compile_definitions(
	cugo
	PRIVATE
	${CUGO_DEFINITIONS}
)

target_include_directories(cugo PUBLIC "${CMAKE_SOURCE_DIR}/src")
set_property(TARGET cugo PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET cugo PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cugo PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (BUILD_CUGO_SHARED AND NOT WIN32)
	target_compile_options(cugo PRIVATE "-fvisibility=hidden")
endif()

install(
	TARGETS cugo 
	LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(
	DIRECTORY ${CUGO_INCLUDE_DIR}/
	DESTINATION "include/cugo"
	FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
install(
	DIRECTORY ${CMAKE_SOURCE_DIR}/src/
	DESTINATION "include/cugo"
	FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

unset(BUILD_CUGO_SHARED CACHE)
