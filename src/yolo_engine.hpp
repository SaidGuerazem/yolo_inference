#ifndef YOLO_ENGINE_HPP
#define YOLO_ENGINE_HPP

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime.h>

// Forward declaration of CUDA kernel
extern "C" void launch_preprocess_kernel(
    const uint8_t* d_input,
    float* d_output,
    int width,
    int height,
    cudaStream_t stream
);

// Detection structure
struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

// Simple TensorRT Logger
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

class YoloEngine {
public:
    YoloEngine(const std::string& engine_path);
    ~YoloEngine();
    
    std::vector<Detection> infer(cv::Mat& frame, float conf_threshold = 0.4f);

private:
    std::vector<Detection> infer_half(cv::Mat& half_frame, float conf_threshold, int x_offset);
    void preprocess(cv::Mat& frame, float* gpu_input);
    std::vector<Detection> postprocess(float* data, cv::Size original_size, float conf_thres, int x_offset);

    // TensorRT members
    Logger gLogger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    
    // CUDA buffers (input and output)
    void* buffers[2];
    uint8_t* d_resized_input;  // GPU buffer for resized uint8 image
    cudaStream_t stream;       // CUDA stream for async operations
    
    // Memory
    size_t input_size;
    size_t output_size;
    float* output_cpu;
    float* pinned_input;  // Pinned host memory for fast transfers
    
    // Preprocessing buffers (reused each frame)
    cv::Mat resized_frame;
    
    // Postprocessing buffers (reused to avoid allocations)
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<int> nms_indices;
    
    // Model dimensions
    int input_w;
    int input_h;
    
    // Performance mode
    bool use_gpu_preprocess;  // Use CUDA kernel for preprocessing
};

#endif // YOLO_ENGINE_HPP
