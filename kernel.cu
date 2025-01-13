#include "kernel.h"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#define AT(x,y) ((x)+(y)*width) 


// Host function to initialize CUDA and select the best device
void initializeCUDA() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Number of CUDA devices: %d\n", deviceCount);

    int bestDevice = 0;
    int maxPerformance = 0;

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %.2f GB\n", (float)prop.totalGlobalMem / (1024 * 1024 * 1024));
        printf("  Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);

        int performance = prop.multiProcessorCount * prop.clockRate;
        if (performance > maxPerformance) {
            bestDevice = i;
            maxPerformance = performance;
        }
    }

    printf("Selecting device %d as the best device.\n", bestDevice);
    cudaSetDevice(bestDevice);

    cudaDeviceProp bestProp;
    cudaGetDeviceProperties(&bestProp, bestDevice);
    printf("Using device: %s\n", bestProp.name);
}

template<BayerPattern pattern>
__global__ void demosaicBayer(const uchar1* bayer, uchar3* output, int width, int height) {

    // Compute the thread's position in the image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure thread is within image bounds
    if (x >= width || y >= height || x < 0 || y < 0) {
        return;
    }
    int x_shift, y_shift;
    switch (pattern) {
        case RGGB:
            x_shift = 0;
            y_shift = 0;
            break;
        case BGGR:
            x_shift = 1;
            y_shift = 1;
            break;
        case GRBG:
            x_shift = 1;
            y_shift = 0;
            break;
        case GBRG:
            x_shift = 0;
            y_shift = 1;
            break;
    }

    uint8_t r;
    uint8_t g;
    uint8_t b;
    
    uint16_t tile[3][3];
    
    if (x < (width-1) && y < (height-1) && x >= 1 && y >= 1) { // thread is not on the border of image
        for (int i=-1; i<2; i++) {
            for (int j=-1; j<2; j++) {
                tile[i+1][j+1] = bayer[AT(x+i,y+j)].x;
            }
        }
    } else {

        for (int i=-1; i<2; i++) {
            for (int j=-1; j<2; j++) {
                int x_idx = x+i;
                if (x_idx < 0) x_idx = x+1;
                if (x_idx >= width) x_idx = x-1;
                int y_idx = y+j;
                if (y_idx < 0) y_idx = y+1;
                if (y_idx >= height) y_idx = y-1;
                tile[i+1][j+1] = bayer[AT(x+i,y+j)].x;
            }
        }
    }
    
    // shift patterns so they match
    int x_pos = (x+x_shift) % 2;
    int y_pos = (y+y_shift) % 2;
    // Determine the color channel for this pixel in the Bayer pattern
    if ((y_pos == 0) && (x_pos == 0)) { // Red pixel (RGGB pattern)
        r = tile[1][1];
        g = (tile[0][1] + tile[1][0] + tile[2][1] + tile[1][2]) / 4;            // Average of left, right, top, and bottom
        b = (tile[0][0] + tile[0][2] + tile[2][0] + tile[2][2]) / 4;            // Average of corners
    } else if ((y_pos == 0) && (x_pos == 1)) { // Green pixel on red row
        r = (tile[0][1] + tile[2][1]) / 2;                                      // Average of left and right
        g = tile[1][1];
        b = (tile[1][0] + tile[1][2]) / 2;                                      // Average of Top and Bottom
    } else if ((y_pos == 1) && (x_pos == 0)) { // Green pixel on blue row
        b = (tile[0][1] + tile[2][1]) / 2;                                      // Average of left and right
        g = tile[1][1];
        r = (tile[1][0] + tile[1][2]) / 2;                                      // Average of Top and Bottom
    } else { // Blue pixel
        b = tile[1][1];
        g = (tile[0][1] + tile[1][0] + tile[2][1] + tile[1][2]) / 4;            // Average of left, right, top, and bottom
        r = (tile[0][0] + tile[0][2] + tile[2][0] + tile[2][2]) / 4;            // Average of corners
    }

    // Store the resulting RGB values in the output image
    int idx = y * width + x;
    output[idx].x = r;
    output[idx].y = g;
    output[idx].z = b;

    // attempt at coalescing global writes
    // it works, but it's slower than the naive write
    // __shared__ uint8_t out[2][3*128];
    // out[threadIdx.y][threadIdx.x*3] = r;
    // out[threadIdx.y][threadIdx.x*3+1] = g;
    // out[threadIdx.y][threadIdx.x*3+2] = b;
    // __syncthreads();
    // uint8_t *output_u8 = (uint8_t*) output;
    // for (int i=threadIdx.y; i<2; i+=2) {
    //     for (int j=threadIdx.x; j<(3*128); j+=128) {
    //         int xt = blockIdx.x * blockDim.x ;
    //         int yt = blockIdx.y * blockDim.y + i;
    //         output_u8[ 3*AT(xt,yt)+j ] = out[i][j]; 
    //     }
    // }
}
// Host function to launch the kernel
void demosaicImage(const unsigned char* bayerImage, unsigned char* outputImage, int width, int height, BayerPattern pattern) {
    // Allocate device memory
    unsigned char *d_bayer, *d_output;
    size_t bayerSize = width * height * sizeof(unsigned char);
    size_t outputSize = width * height * 3 * sizeof(unsigned char);

    cudaMalloc(&d_bayer, bayerSize);
    cudaMalloc(&d_output, outputSize);

    // Copy input data to device
    cudaMemcpy(d_bayer, bayerImage, bayerSize, cudaMemcpyHostToDevice);

    // Configure block and grid sizes
    dim3 blockSize(256, 1);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    printf("Launching kernel!\n");
    switch (pattern) {
        case BayerPattern::RGGB:
            demosaicBayer<RGGB><<<gridSize, blockSize>>>((uchar1*)d_bayer, (uchar3*)d_output, width, height);
            break;
        case BayerPattern::BGGR:
            demosaicBayer<BGGR><<<gridSize, blockSize>>>((uchar1*)d_bayer, (uchar3*)d_output, width, height);
            break;
        case BayerPattern::GRBG:
            demosaicBayer<GRBG><<<gridSize, blockSize>>>((uchar1*)d_bayer, (uchar3*)d_output, width, height);
            break;
        case BayerPattern::GBRG:
            demosaicBayer<GBRG><<<gridSize, blockSize>>>((uchar1*)d_bayer, (uchar3*)d_output, width, height);
            break;
    }
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    // Copy the output data back to host
    cudaMemcpy(outputImage, d_output, outputSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_bayer);
    cudaFree(d_output);
}
