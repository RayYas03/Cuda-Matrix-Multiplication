
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void matrixMultiplication(float* A, float* B, float* C, int rowsA, int colsA_rowsB, int colsB) {
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int k = 0; k < colsA_rowsB; ++k) {
            sum += A[row * colsA_rowsB + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

int main() {
    int rowsA = 1000;
    int colsA_rowsB = 1000;
    int colsB = 1000;

    size_t sizeA = rowsA * colsA_rowsB * sizeof(float);
    size_t sizeB = colsA_rowsB * colsB * sizeof(float);
    size_t sizeC = rowsA * colsB * sizeof(float);

    float* hostA = (float*)malloc(sizeA);
    float* hostB = (float*)malloc(sizeB);
    float* hostC = (float*)malloc(sizeC);

    for (int i = 0; i < rowsA * colsA_rowsB; i++) {
        hostA[i] = rand() / (float)RAND_MAX;
    }

    for (int i = 0; i < colsA_rowsB * colsB; i++) {
        hostB[i] = rand() / (float)RAND_MAX;
    }

    float* deviceA;
    float* deviceB;
    float* deviceC;
    cudaMalloc(&deviceA, sizeA);
    cudaMalloc(&deviceB, sizeB);
    cudaMalloc(&deviceC, sizeC);

    cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((colsB + blockDim.x - 1) / blockDim.x, (rowsA + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, NULL);

    matrixMultiplication<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, rowsA, colsA_rowsB, colsB);

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Elapsed time on GPU: %f ms\n", milliseconds);

    cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        int numProcessors = deviceProp.multiProcessorCount;
        printf("Device %d: Number of Processors: %d\n", i, numProcessors);
    }

    free(hostA);
    free(hostB);
    free(hostC);
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return 0;
}
