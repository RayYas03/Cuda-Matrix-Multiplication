
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void MatrixMulKernel(float* M, float* N, float* P, int M_rows, int M_cols_N_rows, int N_cols) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < M_rows && Col < N_cols) {
        float Pvalue = 0;
        for (int k = 0; k < M_cols_N_rows; ++k) {
            Pvalue += M[Row * M_cols_N_rows + k] * N[k * N_cols + Col];
        }
        P[Row * N_cols + Col] = Pvalue;
    }
}

int main() {
    int M_rows = 1000;
    int M_cols_N_rows = 1000;
    int N_cols = 1000;

    size_t size_M = M_rows * M_cols_N_rows * sizeof(float);
    size_t size_N = M_cols_N_rows * N_cols * sizeof(float);
    size_t size_P = M_rows * N_cols * sizeof(float);

    float* h_M = (float*)malloc(size_M);
    float* h_N = (float*)malloc(size_N);
    float* h_P = (float*)malloc(size_P);

    for (int i = 0; i < M_rows * M_cols_N_rows; i++) {
        h_M[i] = rand() / (float)RAND_MAX;
    }

    for (int i = 0; i < M_cols_N_rows * N_cols; i++) {
        h_N[i] = rand() / (float)RAND_MAX;
    }

    float* d_M;
    float* d_N;
    float* d_P;
    cudaMalloc(&d_M, size_M);
    cudaMalloc(&d_N, size_N);
    cudaMalloc(&d_P, size_P);

    cudaMemcpy(d_M, h_M, size_M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size_N, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16); // Adjust the block dimensions as needed
    dim3 dimGrid((N_cols + dimBlock.x - 1) / dimBlock.x, (M_rows + dimBlock.y - 1) / dimBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, NULL);

    MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, M_rows, M_cols_N_rows, N_cols);

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("The elapsed time in GPU was: %f ms\n", milliseconds);

    cudaMemcpy(h_P, d_P, size_P, cudaMemcpyDeviceToHost);

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        int numProcessors = deviceProp.multiProcessorCount;
        printf("Device %d: Number of Processors: %d\n", i, numProcessors);
    }

    free(h_M);
    free(h_N);
    free(h_P);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}
