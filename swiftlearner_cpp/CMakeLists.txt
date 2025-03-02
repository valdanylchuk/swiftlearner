cmake_minimum_required(VERSION 3.15)

project(SwiftLearnerCPP VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type (you can choose Debug, Release, RelWithDebInfo, MinSizeRel)
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g") # Debugging symbols
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math") # Optimized
endif()

message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Compiler Path: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")

find_package(ZLIB REQUIRED)

file(GLOB_RECURSE SL_SOURCES "src/*.cpp" "src/*/*.cpp")

add_library(swiftlearner SHARED ${SL_SOURCES})

# Headers mapping for a potential installation
target_include_directories(swiftlearner PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include> #Adjust if you install headers
)

target_link_libraries(swiftlearner PRIVATE ZLIB::ZLIB)

add_subdirectory(test)