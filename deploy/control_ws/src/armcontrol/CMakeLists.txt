cmake_minimum_required(VERSION 3.8)
project(armcontrol)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 明确告诉 CMake 使用哪个 Boost Python 组件
set(Boost_NO_BOOST_CMAKE ON)
set(Boost_NO_SYSTEM_PATHS OFF)
set(BOOST_PYTHON_SUFFIX "310")  

# 对于 eigenpy 特定的设置
set(eigenpy_FOUND TRUE)
set(EIGENPY_FOUND TRUE)

# 手动设置 Boost Python 组件
set(Boost_PYTHON_FOUND TRUE)
set(Boost_PYTHON3_FOUND TRUE)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(logger REQUIRED)#日志库依赖，jingyi
find_package(visualization_msgs REQUIRED)#gripper test
find_package(tf2_ros REQUIRED)           # 右臂换到左臂下面，jingyi
find_package(geometry_msgs REQUIRED)     # 右臂换到左臂下面，jingyi
find_package(tf2 REQUIRED)               # 右臂换到左臂下面，jingyi

include_directories(include
  ${EIGEN3_INCLUDE_DIRS}
  ${logger_INCLUDE_DIRS}#将logger的include目录添加到include路径中，jingyi
)
add_executable(arm_control_node src/arm_control_node.cpp src/path_smoother.cpp src/socketcan.cpp)
ament_target_dependencies(arm_control_node rclcpp std_msgs sensor_msgs ament_index_cpp visualization_msgs tf2_ros geometry_msgs tf2)
target_link_libraries(arm_control_node
  nlohmann_json::nlohmann_json
  logger::logger#将日志库链接到arm_control_node，jingyi
  atomic
)

# ---- 通信方式宏定义（按架构自动选择，可手动覆盖）----
#用法示例：
# 自动按架构选择：colcon build
# 强制 CAN：colcon build --cmake-args -DARMCOMM=CAN
# 强制 SERIAL：colcon build --cmake-args -DARMCOMM=SERIAL

# 选项：AUTO（默认，按架构推断）、CAN、SERIAL
set(ARMCOMM "AUTO" CACHE STRING "Communication backend: AUTO, CAN, SERIAL")
set_property(CACHE ARMCOMM PROPERTY STRINGS AUTO CAN SERIAL)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" SYS_ARCH_LOWER)
if(ARMCOMM STREQUAL "AUTO")
  if(SYS_ARCH_LOWER MATCHES "^(x86_64|amd64|i686|x86)$")
    set(ARMCOMM_EFFECTIVE "SERIAL")
  elseif(SYS_ARCH_LOWER MATCHES "^(aarch64|arm64|armv7|armv8|arm)$")
    set(ARMCOMM_EFFECTIVE "CAN")
  else()
    # 未识别架构时，默认走串口
    set(ARMCOMM_EFFECTIVE "SERIAL")
  endif()
else()
  set(ARMCOMM_EFFECTIVE "${ARMCOMM}")
endif()

message(STATUS "armcontrol: ARMCOMM=${ARMCOMM} -> ${ARMCOMM_EFFECTIVE} (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")

if(ARMCOMM_EFFECTIVE STREQUAL "CAN")
  target_compile_definitions(arm_control_node PRIVATE USE_CAN_COMMUNICATION)
elseif(ARMCOMM_EFFECTIVE STREQUAL "SERIAL")
  target_compile_definitions(arm_control_node PRIVATE USE_SERIAL_COMMUNICATION)
else()
  message(FATAL_ERROR "Invalid ARMCOMM value: ${ARMCOMM}")
endif()
# ---- 通信方式宏定义结束 ----

install(TARGETS
  arm_control_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
