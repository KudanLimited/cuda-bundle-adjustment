# Load the debug and release variables
get_filename_component(_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
file(GLOB DATA_FILES "${_DIR}/libdeflate-*-data.cmake")

foreach(f ${DATA_FILES})
    include(${f})
endforeach()

# Create the targets for all the components
foreach(_COMPONENT ${libdeflate_COMPONENT_NAMES} )
    if(NOT TARGET ${_COMPONENT})
        add_library(${_COMPONENT} INTERFACE IMPORTED)
        conan_message(STATUS "Conan: Component target declared '${_COMPONENT}'")
    else()
        message(WARNING "Component target name '${_COMPONENT}' already exists.")
    endif()
endforeach()

if(NOT TARGET libdeflate::libdeflate)
    add_library(libdeflate::libdeflate INTERFACE IMPORTED)
    conan_message(STATUS "Conan: Target declared 'libdeflate::libdeflate'")
else()
    message(WARNING "Target name 'libdeflate::libdeflate' already exists.")
endif()
# Load the debug and release library finders
get_filename_component(_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
file(GLOB CONFIG_FILES "${_DIR}/libdeflate-Target-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()