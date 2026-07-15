cmake_minimum_required(VERSION 3.22)
project(logger)
add_library(logger SHARED Logger.cpp)

target_include_directories(logger PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:./>
)

target_link_libraries(logger PUBLIC spdlog fmt)  # 假设用的是spdlog

install(TARGETS logger
    EXPORT logger
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION include/logger
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

install(EXPORT logger
    FILE loggerConfig.cmake
    NAMESPACE logger::
    DESTINATION lib/cmake/logger
)