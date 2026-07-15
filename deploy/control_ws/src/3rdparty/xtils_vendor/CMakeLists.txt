cmake_minimum_required(VERSION 3.14)
project(xtils_vendor)

# find dependencies
find_package(ament_cmake REQUIRED)

include(FetchContent)

FetchContent_Declare(
  xtils
  GIT_REPOSITORY https://github.com/Lingzo/xtils.git
  GIT_TAG release
)

FetchContent_MakeAvailable(xtils)

# export dependencies
ament_export_dependencies(xtils)
ament_package(CONFIG_EXTRAS targets-extras.cmake.in)
