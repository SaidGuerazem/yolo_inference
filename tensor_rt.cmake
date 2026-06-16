
# 1. Provide an option to pass custom TensorRT installation paths
set(TENSORRT_ROOT "" CACHE PATH "Path to the TensorRT installation directory")

# 2. Define search hints based on typical layouts
if(TENSORRT_ROOT)
    set(TRT_INCLUDE_HINTS "${TENSORRT_ROOT}/include")
    set(TRT_LIB_HINTS "${TENSORRT_ROOT}/lib")
else()
    set(TRT_INCLUDE_HINTS "/usr/include/x86_64-linux-gnu" "/usr/include/aarch64-linux-gnu" "/usr/local/cuda/include")
    set(TRT_LIB_HINTS "/usr/lib/x86_64-linux-gnu" "/usr/lib/aarch64-linux-gnu" "/usr/local/cuda/lib64")
endif()

# 3. Find Core Elements
find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h HINTS ${TRT_INCLUDE_HINTS})
find_library(TENSORRT_LIBRARY NAMES nvinfer HINTS ${TRT_LIB_HINTS})

# Optional auxiliary libraries
find_library(TENSORRT_PLUGIN_LIBRARY NAMES nvinfer_plugin HINTS ${TRT_LIB_HINTS})
find_library(TENSORRT_ONNX_LIBRARY NAMES nvonnxparser HINTS ${TRT_LIB_HINTS})

# Verify basic requirements
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT could not be found. Please set -DTENSORRT_ROOT=/path/to/tensorrt")
endif()

# 4. Create the modern TensorRT Interface Target
# Note: UNKNOWN IMPORTED allows linking both static or shared libs safely
add_library(TensorRT::TensorRT UNKNOWN IMPORTED GLOBAL)

# Set the include directories property
set_target_properties(TensorRT::TensorRT PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
    IMPORTED_LOCATION "${TENSORRT_LIBRARY}"
)

# 5. Attach optional component libraries to the primary interface target
if(TENSORRT_PLUGIN_LIBRARY)
    target_link_libraries(TensorRT::TensorRT INTERFACE "${TENSORRT_PLUGIN_LIBRARY}")
endif()

if(TENSORRT_ONNX_LIBRARY)
    target_link_libraries(TensorRT::TensorRT INTERFACE "${TENSORRT_ONNX_LIBRARY}")
endif()