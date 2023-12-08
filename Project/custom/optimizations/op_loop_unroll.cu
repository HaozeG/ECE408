#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

// tuning with restrict and loop unroll
__global__ void conv_forward_kernel(float *output, const float * __restrict__ input, const float * __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int b = blockIdx.x;
    int m = blockIdx.z;
    int h = blockIdx.y;
    int w = threadIdx.x;
    float sum = 0.0f;
    for (int c = 0; c < C; c++) {
        // #pragma unroll
        for (int p = 0; p < K; p++) {
            if (K == 7) {
                sum += in_4d(b, c, h*S + p, w*S + 0) * mask_4d(m, c, p, 0)
                + in_4d(b, c, h*S + p, w*S + 1) * mask_4d(m, c, p, 1)
                + in_4d(b, c, h*S + p, w*S + 2) * mask_4d(m, c, p, 2)
                + in_4d(b, c, h*S + p, w*S + 3) * mask_4d(m, c, p, 3)
                + in_4d(b, c, h*S + p, w*S + 4) * mask_4d(m, c, p, 4)
                + in_4d(b, c, h*S + p, w*S + 5) * mask_4d(m, c, p, 5)
                + in_4d(b, c, h*S + p, w*S + 6) * mask_4d(m, c, p, 6);
            }
            if (K == 3) {
                sum += in_4d(b, c, h*S + p, w*S + 0) * mask_4d(m, c, p, 0)
                + in_4d(b, c, h*S + p, w*S + 1) * mask_4d(m, c, p, 1)
                + in_4d(b, c, h*S + p, w*S + 2) * mask_4d(m, c, p, 2);
            }
            // for (int q = 0; q < K; q++) {
            // }
        }
    }
    out_4d(b, m, h, w) = sum;

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    int size_input = sizeof(float) * B * C * H * W;
    int size_mask = sizeof(float) * M * C * K * K;
    int size_output = sizeof(float) * B * M * H_out * W_out;
    // Allocate memory and copy over the relevant data structures to the GPU
    cudaMalloc((void **)device_input_ptr, size_input);
    cudaMalloc((void **)device_mask_ptr, size_mask);
    cudaMalloc((void **)device_output_ptr, size_output);

    cudaMemcpy(*device_input_ptr, host_input, size_input, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, size_mask, cudaMemcpyHostToDevice);
    cudaMemset(*device_output_ptr, 0.0f, size_output);
    
    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.
    // std::cout<<"no error in prolog"<<std::endl;
    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
   
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    // Set the kernel dimensions and call the kernel
    dim3 dimGrid(B, H_out, M);
    dim3 dimBlock(W_out, 1, 1);

    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);

    // std::cout<<"no error calling kernel"<<std::endl;
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    int outsize = sizeof(float) * H_out * W_out * B * M;
    // Copy the output back to host
    cudaMemcpy(host_output, device_output, outsize, cudaMemcpyDeviceToHost);
   
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);

    // std::cout<<"no error calling epilog"<<std::endl;
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
