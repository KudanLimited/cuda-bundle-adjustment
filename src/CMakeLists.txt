cmake_minimum_required(VERSION 3.16)

project(cugo LANGUAGES CXX CUDA)

set(CUGO_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

find_package(Eigen3 REQUIRED)

if (DEBUG_CUGO)
	set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")
else()
	set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-lineinfo")
endif()

set(CUGO_COMPILE_FLAGS "")
set(CUGO_DEFINITIONS "")

if (CMAKE_COMPILER_IS_GNUCXX)
	list(APPEND CUGO_COMPILE_FLAGS -O3 -Wall)
endif()

if (NOT CUDA_TOOLKIT_LIB_TYPE)
	set(CUDA_TOOLKIT_LIB_TYPE "static")
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
	list(APPEND CUGO_DEFINITIONS USE_FLOAT32)
endif()
if (USE_TOOLKIT_10_2)
	list(APPEND CUGO_DEFINITIONS USE_TOOLKIT_10_2)
endif()
if (USE_ZERO_COPY)
	list(APPEND CUGO_DEFINITIONS USE_ZERO_COPY)
endif()

message("\nCompiler definitions:    ${CUGO_DEFINITIONS}")
message("Compiler options:        ${CUDA_COMPILE_FLAGS}")
message("CUDA toolkit build type: ${CUDA_TOOLKIT_LIB_TYPE}")
message("CUDA compiler flags:     ${CMAKE_CUDA_FLAGS}\n")

file(GLOB_RECURSE srcs ./*.cpp ./*.h ./*.hpp ./*.cu ${CUGO_INCLUDE_DIR}/*.h)

if (BUILD_CUGO_SHARED) 
	message("Building CUGO as shared library....")
	add_library(cugo SHARED ${srcs})
else()
	message("Building CUGO as static library....")
	add_library(cugo STATIC ${srcs})
endif()

if (CUDA_TOOLKIT_LIB_TYPE STREQUAL "shared" OR WIN32)
	set(CUDA_LIBS 
		CUDA::cusparse 
		CUDA::cusolver
		CUDA::cublas
		CUDA::cublasLt 
		CUDA::cudart
	)
elseif(CUDA_TOOLKIT_LIB_TYPE STREQUAL "static")
	set(CUDA_LIBS 
		CUDA::cusparse_static 
		CUDA::cusolver_static 
		CUDA::cublas_static 
		CUDA::cublasLt_static 
		CUDA::cudart_static
		# The toolkit contains no target for libmetis for some reason.
		${CUDAToolkit_LIBRARY_DIR}/libmetis_static.a
	)
else()
	message(FATAL_ERROR "Unknown value for CUDA_TOOLKIT_LIB_TYPE: ${CUDA_TOOLKIT_LIB_TYPE}")
endif()

target_include_directories(
	cugo 
	PUBLIC
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
	"$<$<NOT:$<PLATFORM_ID:Windows>>:-fPIC>"
)
target_compile_definitions(
	cugo
	PUBLIC
	${CUGO_DEFINITIONS}
)

target_include_directories(cugo PUBLIC "${CMAKE_SOURCE_DIR}/src")
set_property(TARGET cugo PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET cugo PROPERTY CUDA_SEPARABLE_COMPILATION ON)

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