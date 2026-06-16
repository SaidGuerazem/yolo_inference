cmake_minimum_required(VERSION 3.10)

project(
    yolo_inference
    VERSION 0.0.1
    DESCRIPTION "YOLO TensorRT detector with integrated ground-target pose estimation"
    LANGUAGES C CXX CUDA
)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE Release CACHE STRING "build type" FORCE)

# Jetson Orin / Orin Nano = SM 8.7. Override at configure time if needed:
# colcon build --cmake-args -DCMAKE_CUDA_ARCHITECTURES=87
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "CUDA enabled. Toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

include(tensor_rt.cmake)
include(cmake/gstreamer.cmake)

# ROS 2
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(px4_msgs REQUIRED)

add_executable(yolo_cam_optimized
    ${CMAKE_CURRENT_LIST_DIR}/src/main_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/yolo_engine.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/preprocess_kernel.cu
)

target_include_directories(yolo_cam_optimized
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(yolo_cam_optimized
    rclcpp
    std_msgs
    geometry_msgs
    px4_msgs
)

target_link_libraries(yolo_cam_optimized
    CUDA::cudart
    TensorRT::TensorRT
    ${OpenCV_LIBS}
)

install(TARGETS yolo_cam_optimized
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
)

ament_package()
