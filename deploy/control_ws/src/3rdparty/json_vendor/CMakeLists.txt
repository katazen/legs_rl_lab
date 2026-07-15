cmake_minimum_required(VERSION 3.14)
project(json_vendor)

# Allow sub-projects with older CMake version requirements
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# Set CMake policy to allow older versions in sub-projects
cmake_policy(SET CMP0048 NEW)

# find dependencies
find_package(ament_cmake REQUIRED)

include(FetchContent)

# pass to sub project
set(JSON_MultipleHeaders
    ON
    CACHE BOOL "" FORCE)

set(JSON_Install
    ON
    CACHE BOOL "" FORCE)
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.10.5)
FetchContent_MakeAvailable(json)

# export dependencies
ament_export_dependencies(nlohmann_json)
ament_package(CONFIG_EXTRAS targets-extras.cmake.in)
