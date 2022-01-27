# Load the debug and release variables
get_filename_component(_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
file(GLOB DATA_FILES "${_DIR}/g2o-*-data.cmake")

foreach(f ${DATA_FILES})
    include(${f})
endforeach()

# Create the targets for all the components
foreach(_COMPONENT ${g2o_COMPONENT_NAMES} )
    if(NOT TARGET ${_COMPONENT})
        add_library(${_COMPONENT} INTERFACE IMPORTED)
        conan_message(STATUS "Conan: Component target declared '${_COMPONENT}'")
    else()
        message(WARNING "Component target name '${_COMPONENT}' already exists.")
    endif()
endforeach()

if(NOT TARGET G2O::G2O)
    add_library(G2O::G2O INTERFACE IMPORTED)
    conan_message(STATUS "Conan: Target declared 'G2O::G2O'")
else()
    message(WARNING "Target name 'G2O::G2O' already exists.")
endif()
# Load the debug and release library finders
get_filename_component(_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
file(GLOB CONFIG_FILES "${_DIR}/g2o-Target-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()