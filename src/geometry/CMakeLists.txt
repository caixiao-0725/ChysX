cmake_minimum_required(VERSION 3.18)

# Project name: geometry
project(geometry LANGUAGES CXX CUDA)

# Organize targets into folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# C++ standard options (adjust as needed)
set(GEOMETRY_CPP_STANDARD "17" CACHE STRING "C++ standard (17/20/23/26)")
set_property(CACHE GEOMETRY_CPP_STANDARD PROPERTY STRINGS 17 20 23 26)

# Target name
set(TARGET_NAME geometry)

# Set C++ standard
set(CMAKE_CXX_STANDARD ${GEOMETRY_CPP_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD ${GEOMETRY_CPP_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)



# Source files
file(GLOB CUDAAPI_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/include/geometry/*.h"
    "${CMAKE_SOURCE_DIR}/include/geometry/*.cuh"
    "${CMAKE_SOURCE_DIR}/src/geometry/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/geometry/*.cu"
)
# Create the static library
add_library(${TARGET_NAME} STATIC ${CUDAAPI_SOURCES})

# Link libraries
target_link_libraries(${TARGET_NAME} PRIVATE cudaAPI)

# Include directories
target_include_directories(${TARGET_NAME} PUBLIC  ${CMAKE_SOURCE_DIR}/include/cudaAPI)

# MSVC settings
if(MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/MP /W4 /WX /utf-8> /wd4505)
endif()