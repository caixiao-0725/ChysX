# cmake 

cmake_minimum_required(VERSION 3.18)

# Project name: ChysX
project(ChysX LANGUAGES CXX CUDA)

# Organize targets into folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Library options
option(CHYSX_BUILD_TESTS "Build tests for chysx library" ON)
option(CHYSX_BUILD_SAMPLES "Build samples for chysx library" OFF)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


# Include directories
include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/foundation)


add_subdirectory(src)

# Optional: Add tests if enabled
if(CHYSX_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Optional: Add examples if enabled
if(CHYSX_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Install headers
file(GLOB_RECURSE FOUNDATION_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/foundation/*.h")


source_group("Header Files\\foundation" FILES ${FOUNDATION_HEADERS})