cmake_minimum_required(VERSION 3.18)

# Project name: physics
project(physics LANGUAGES CXX CUDA)

# Organize targets into folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# C++ standard options (adjust as needed)
set(PHYSICS_CPP_STANDARD "17" CACHE STRING "C++ standard (17/20/23/26)")
set_property(CACHE PHYSICS_CPP_STANDARD PROPERTY STRINGS 17 20 23 26)

# Target name
set(TARGET_NAME physics)

# Set C++ standard
set(CMAKE_CXX_STANDARD ${PHYSICS_CPP_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD ${PHYSICS_CPP_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Source files - currently only timeStep has implementation
file(GLOB PHYSICS_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/physics/**/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/physics/**/*.cu"
)

# Header files
file(GLOB PHYSICS_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/include/physics/**/*.h"
    "${CMAKE_SOURCE_DIR}/include/physics/**/*.cuh"
)

# Create the static library
add_library(${TARGET_NAME} STATIC ${PHYSICS_SOURCES} ${PHYSICS_HEADERS})

# Include directories
target_include_directories(${TARGET_NAME} PUBLIC 
    ${CMAKE_SOURCE_DIR}/include/physics
    ${CMAKE_SOURCE_DIR}/include/foundation
    ${CMAKE_SOURCE_DIR}/include/cudaAPI
)

# Link dependencies
target_link_libraries(${TARGET_NAME} PUBLIC 
    cudaAPI
)

# MSVC settings
if(MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/MP /W4 /WX /utf-8> /wd4505)
endif()

