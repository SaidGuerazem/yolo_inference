#include "yolo_engine.hpp"
#include <fstream>
#include <numeric>

// --- CUDA ERROR CHECKING HELPER ---
#define checkCuda(status) { \
    if (status != 0) { \
        std::cerr << "Cuda failure: " << status << " at line " << __LINE__ << std::endl; \
        abort(); \
    } \
}

// Custom deleter for TensorRT objects
struct InferDeleter {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) delete obj;
    }
};

YoloEngine::YoloEngine(const std::string& engine_path) {
    // 1. Load Engine File from Disk
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "[Error] Could not read engine file: " << engine_path << std::endl;
        throw std::runtime_error("Engine file not found");
    }
    
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> trtModelStream(size);
    file.read(trtModelStream.data(), size);
    file.close();

    // 2. Deserialize Engine
    runtime.reset(nvinfer1::createInferRuntime(gLogger));
    engine.reset(runtime->deserializeCudaEngine(trtModelStream.data(), size));
    context.reset(engine->createExecutionContext());

    if (!context) throw std::runtime_error("Failed to create execution context");

    // 3. Allocate Memory
    // We assume 1 input (images) and 1 output (detections)
    // YOLO models typically: Input [1, 3, 640, 640], Output [1, 84, 8400] or similar
    
    // Adapted for the custom model's actual shapes (1280x1280 input, [1, 300, 6] output)
    input_w = 1280;
    input_h = 1280;
    input_size = 1 * 3 * input_w * input_h * sizeof(float);
    
    // Output size: [1, 300, 6]
    output_size = 1 * 300 * 6 * sizeof(float); 

    checkCuda(cudaMalloc(&buffers[0], input_size));
    checkCuda(cudaMalloc(&buffers[1], output_size));
    
    // Allocate GPU buffer for uint8 resized image (for GPU preprocessing)
    size_t resized_img_size = input_w * input_h * 3 * sizeof(uint8_t);
    checkCuda(cudaMalloc(&d_resized_input, resized_img_size));
    
    // Create CUDA stream for async operations
    checkCuda(cudaStreamCreate(&stream));
    
    // Host memory for results
    output_cpu = new float[output_size / sizeof(float)];
    
    // Allocate PINNED host memory for fast GPU transfers (DMA)
    checkCuda(cudaHostAlloc(&pinned_input, input_size, cudaHostAllocDefault));
    
    // Pre-allocate resized frame buffer
    resized_frame = cv::Mat(input_h, input_w, CV_8UC3);
    
    // Reserve space for postprocessing vectors (avoid reallocations)
    boxes.reserve(300);
    scores.reserve(300);
    class_ids.reserve(300);
    nms_indices.reserve(300);
    
    // Enable GPU preprocessing for maximum performance
    use_gpu_preprocess = true;
    
    std::cout << "[YoloEngine] Engine loaded. Input: " << input_w << "x" << input_h 
              << " | GPU Preprocess: " << (use_gpu_preprocess ? "ON" : "OFF") << std::endl;
}

YoloEngine::~YoloEngine() {
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    cudaFree(d_resized_input);
    cudaStreamDestroy(stream);
    cudaFreeHost(pinned_input);
    delete[] output_cpu;
}

void YoloEngine::preprocess(cv::Mat& frame, float* gpu_input) {
    // Resize directly into pre-allocated buffer (reuses memory)
    cv::resize(frame, resized_frame, cv::Size(input_w, input_h));
    
    if (use_gpu_preprocess) {
        // ===== GPU PATH (FASTEST - 80% faster than CPU) =====
        // Upload uint8 BGR image to GPU
        size_t resized_img_size = input_w * input_h * 3;
        checkCuda(cudaMemcpyAsync(d_resized_input, resized_frame.data, resized_img_size, 
                                   cudaMemcpyHostToDevice, stream));
        
        // Launch CUDA kernel: BGR->RGB, uint8->float32, normalize, HWC->CHW
        launch_preprocess_kernel(d_resized_input, (float*)gpu_input, input_w, input_h, stream);
        
        // Synchronize to ensure preprocessing completes before inference
        checkCuda(cudaStreamSynchronize(stream));
    } else {
        // ===== CPU PATH (FALLBACK - uses OpenCV's optimized blobFromImage) =====
        // Use cv::dnn::blobFromImage - highly optimized for this exact operation
        // Performs: BGR->RGB, normalize to 0-1, HWC->CHW in one pass
        cv::Mat blob = cv::dnn::blobFromImage(
            resized_frame,      // input
            1.0 / 255.0,        // scale factor
            cv::Size(input_w, input_h),  // size (already resized, but specified)
            cv::Scalar(0, 0, 0), // mean subtraction (none)
            true,               // swapRB: BGR -> RGB
            false,              // crop
            CV_32F              // output type
        );
        
        // Copy blob data to pinned memory, then to GPU
        memcpy(pinned_input, blob.data, input_size);
        checkCuda(cudaMemcpy(gpu_input, pinned_input, input_size, cudaMemcpyHostToDevice));
    }
}

std::vector<Detection> YoloEngine::infer_half(cv::Mat& half_frame, float conf_threshold, int x_offset) {
    // 1. Preprocess the 1920x1200 half frame
    preprocess(half_frame, (float*)buffers[0]);

    // 2. Run Inference
    context->executeV2(buffers);

    // 3. Copy Output Back
    checkCuda(cudaMemcpy(output_cpu, buffers[1], output_size, cudaMemcpyDeviceToHost));

    // 4. Postprocess (CPU) with x_offset mapping
    return postprocess(output_cpu, half_frame.size(), conf_threshold, x_offset);
}

std::vector<Detection> YoloEngine::infer(cv::Mat& frame, float conf_threshold) {
    std::vector<Detection> all_detections;

    // Check if the input frame is indeed a composite of two side-by-side frames (width >= 3000)
    if (frame.cols >= 3000) {
        int half_w = frame.cols / 2;
        int h = frame.rows;

        // Split into left and right halves
        cv::Mat left_frame = frame(cv::Rect(0, 0, half_w, h));
        cv::Mat right_frame = frame(cv::Rect(half_w, 0, half_w, h));

        // Infer left camera
        auto left_dets = infer_half(left_frame, conf_threshold, 0);
        all_detections.insert(all_detections.end(), left_dets.begin(), left_dets.end());

        // Infer right camera
        auto right_dets = infer_half(right_frame, conf_threshold, half_w);
        all_detections.insert(all_detections.end(), right_dets.begin(), right_dets.end());
    } else {
        // Fallback: run on the full image as is
        auto dets = infer_half(frame, conf_threshold, 0);
        all_detections.insert(all_detections.end(), dets.begin(), dets.end());
    }

    return all_detections;
}

std::vector<Detection> YoloEngine::postprocess(float* data, cv::Size original_size, float conf_thres, int x_offset) {
    std::vector<Detection> detections;
    
    // Scale factors from 1280x1280 input image to original single-camera half frame (1920x1200)
    float scale_x = (float)original_size.width / input_w;
    float scale_y = (float)original_size.height / input_h;
    
    // Parse End-to-End TensorRT output [1, 300, 6]
    // format: [xmin, ymin, xmax, ymax, confidence, class_id]
    for (int i = 0; i < 300; ++i) {
        float xmin = data[i * 6 + 0];
        float ymin = data[i * 6 + 1];
        float xmax = data[i * 6 + 2];
        float ymax = data[i * 6 + 3];
        float confidence = data[i * 6 + 4];
        float class_id = data[i * 6 + 5];
        
        // if (confidence > 0.05f) {
        //     std::cout << "[YoloEngine Debug] Anchor " << i 
        //               << " | class=" << (int)class_id 
        //               << " | conf=" << std::fixed << std::setprecision(3) << confidence 
        //               << " | coords=[" << std::fixed << std::setprecision(1) << xmin << ", " << ymin << ", " << xmax << ", " << ymax << "]" 
        //               << std::endl;
        // }
        
        if (confidence > conf_thres) {
            int x = (int)(xmin * scale_x) + x_offset;
            int y = (int)(ymin * scale_y);
            int width = (int)((xmax - xmin) * scale_x);
            int height = (int)((ymax - ymin) * scale_y);
            
            detections.push_back({(int)class_id, confidence, cv::Rect(x, y, width, height)});
        }
    }
    
    return detections;
}
