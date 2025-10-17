cmake_minimum_required(VERSION 3.18)

# Project name: simulator
project(simulator LANGUAGES CXX CUDA)

# Organize targets into folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# C++ standard options (adjust as needed)
set(SIMULATOR_CPP_STANDARD "17" CACHE STRING "C++ standard (17/20/23/26)")
set_property(CACHE SIMULATOR_CPP_STANDARD PROPERTY STRINGS 17 20 23 26)

# Target name
set(TARGET_NAME simulator)

# Set C++ standard
set(CMAKE_CXX_STANDARD ${SIMULATOR_CPP_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD ${SIMULATOR_CPP_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Source files
file(GLOB SIMULATOR_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/include/simulator/**/*.h"
    "${CMAKE_SOURCE_DIR}/include/simulator/**/*.cuh"
    "${CMAKE_SOURCE_DIR}/include/simulator/*.h"
    "${CMAKE_SOURCE_DIR}/include/simulator/*.cuh"
    "${CMAKE_SOURCE_DIR}/src/simulator/**/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/simulator/**/*.cu"
    "${CMAKE_SOURCE_DIR}/src/simulator/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/simulator/*.cu"
)

# Create the static library
add_library(${TARGET_NAME} STATIC ${SIMULATOR_SOURCES})

# Link libraries
target_link_libraries(${TARGET_NAME} PRIVATE cudaAPI)
target_link_libraries(${TARGET_NAME} PRIVATE geometry)

# Include directories
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include/simulator)
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include/cudaAPI)
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include/geometry)
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include/foundation)

# MSVC settings
if(MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/MP /W4 /WX /utf-8> /wd4505)
endif()