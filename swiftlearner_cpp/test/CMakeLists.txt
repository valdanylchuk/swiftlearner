find_package(GTest REQUIRED)
find_package(ZLIB REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})

file(GLOB_RECURSE SL_TEST_SOURCES "*.cpp" "*/*.cpp")

add_executable(swiftlearner_test ${SL_TEST_SOURCES})
target_link_libraries(swiftlearner_test swiftlearner GTest::gtest GTest::gtest_main ZLIB::ZLIB)

include(GoogleTest)
gtest_discover_tests(swiftlearner_test)

# Copy the entire test/resources directory to the build directory
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/resources"
    DESTINATION "${CMAKE_CURRENT_BINARY_DIR}"
)