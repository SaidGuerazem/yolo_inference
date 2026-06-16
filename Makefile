# Makefile for YOLO TensorRT Integration with CUDA Preprocessing + ROS2

# Compiler settings
CXX = g++
NVCC = nvcc
TARGET = yolo_cam_optimized

# Directories
TENSORRT_DIR = /usr/lib/aarch64-linux-gnu
CUDA_DIR = /usr/local/cuda
ROS2_DIR = /opt/ros/humble

# Compiler flags
CXXFLAGS = -std=c++17 -O3 -Wall -pthread
NVCCFLAGS = -std=c++17 -O3 --compiler-options '-fPIC'

# ROS2 Include paths - dynamically include all subdirectories
ROS2_INCLUDES = $(shell find $(ROS2_DIR)/include -maxdepth 1 -type d | sed 's/^/-I/')

# Include paths
INCLUDES = -I$(CUDA_DIR)/include \
           -I/usr/include/opencv4 \
           -I$(TENSORRT_DIR) \
           $(ROS2_INCLUDES)

# Library paths and libraries
LDFLAGS = -L$(CUDA_DIR)/lib64 \
          -L$(TENSORRT_DIR) \
          -L/usr/lib/aarch64-linux-gnu \
          -L$(ROS2_DIR)/lib

# ROS2 Libraries - all required for rclcpp and vision_msgs
ROS2_LIBS = -lrclcpp \
            -lrcl \
            -lrcl_yaml_param_parser \
            -lrcutils \
            -lrmw \
            -lrosidl_runtime_c \
            -lrosidl_typesupport_cpp \
            -lrosidl_typesupport_c \
            -lrosidl_typesupport_introspection_cpp \
            -ltracetools \
            -lcv_bridge \
            -lvision_msgs__rosidl_typesupport_cpp \
            -lvision_msgs__rosidl_generator_c \
            -lsensor_msgs__rosidl_typesupport_cpp \
            -lsensor_msgs__rosidl_generator_c \
            -lgeometry_msgs__rosidl_typesupport_cpp \
            -lgeometry_msgs__rosidl_generator_c \
            -lstd_msgs__rosidl_typesupport_cpp \
            -lstd_msgs__rosidl_generator_c \
            -lbuiltin_interfaces__rosidl_typesupport_cpp \
            -lbuiltin_interfaces__rosidl_generator_c

LIBS = -lnvinfer \
       -lnvonnxparser \
       -lcudart \
       -lopencv_core \
       -lopencv_imgproc \
       -lopencv_videoio \
       -lopencv_highgui \
       -lopencv_dnn \
       -lgstreamer-1.0 \
       -lgobject-2.0 \
       -lglib-2.0 \
       -lpthread \
       $(ROS2_LIBS)

# Source files
CPP_SOURCES = main_test.cpp
CU_SOURCES = preprocess_kernel.cu

# Object files
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
CU_OBJECTS = $(CU_SOURCES:.cu=.o)

# Default target
all: $(TARGET)

# Link everything together
$(TARGET): $(CPP_OBJECTS) $(CU_OBJECTS)
	@echo "Linking $(TARGET)..."
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)
	@echo "Build complete! Run with: ./$(TARGET)"

# Compile C++ files
%.o: %.cpp yolo_engine.hpp yolo_engine.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile CUDA files
%.o: %.cu
	@echo "Compiling CUDA kernel $<..."
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(TARGET) $(CPP_OBJECTS) $(CU_OBJECTS)
	@echo "Clean complete!"

# Phony targets
.PHONY: all clean

# Help target
help:
	@echo "YOLO TensorRT + ROS2 Makefile"
	@echo "================================"
	@echo "Targets:"
	@echo "  all    - Build the multi-threaded ROS2 executable (default)"
	@echo "  clean  - Remove all build artifacts"
	@echo "  help   - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make           # Build the project"
	@echo "  make clean     # Clean build files"
	@echo "  ./$(TARGET)    # Run the program"
	@echo ""
	@echo "Requirements:"
	@echo "  - ROS2 (rclcpp, std_msgs)"
	@echo "  - TensorRT, CUDA, OpenCV, GStreamer"
