#include <cuda_runtime.h>
#include <stdint.h>

// CUDA kernel for preprocessing: BGR->RGB, uint8->float32, normalize, HWC->CHW
__global__ void preprocess_kernel(
    const uint8_t* input,  // Input image [H, W, 3] BGR uint8
    float* output,         // Output tensor [3, H, W] RGB float32
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int hwc_idx = (y * width + x) * 3;  // HWC index
    int hw_size = height * width;
    int chw_idx = y * width + x;        // CHW base index
    
    // Read BGR, convert to RGB, normalize to [0,1]
    output[0 * hw_size + chw_idx] = input[hwc_idx + 2] / 255.0f; // R
    output[1 * hw_size + chw_idx] = input[hwc_idx + 1] / 255.0f; // G
    output[2 * hw_size + chw_idx] = input[hwc_idx + 0] / 255.0f; // B
}

// Host function to launch the kernel
extern "C" void launch_preprocess_kernel(
    const uint8_t* d_input,
    float* d_output,
    int width,
    int height,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    preprocess_kernel<<<grid, block, 0, stream>>>(d_input, d_output, width, height);
}
