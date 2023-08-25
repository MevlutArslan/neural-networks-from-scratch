#include "nmatrix.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <cassert>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

__global__ void copy_matrix_kernel(double* source, double* destination, int rows, int cols) {
    int row = (blockIdx.y * blockDim.y) + threadIdx.y;
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;

    int index = row * cols + col;

    if(index < rows * cols) {
        destination[index] = source[index];
    }
}

#ifdef __cplusplus
extern "C" {
#endif
    void copy_matrix_cuda(Matrix* source, Matrix* destination) {
        assert(source != NULL && destination != NULL);
        assert(source->rows == destination->rows && source->columns == destination->columns);

        double* source_gpu;
        double* destination_gpu;
        size_t matrix_size = source->rows * source->columns * sizeof(double);

        cudaMalloc(&source_gpu, matrix_size);
        cudaMalloc(&destination_gpu, matrix_size);

        cudaMemcpy(source_gpu, source->data->elements, matrix_size, cudaMemcpyHostToDevice);

        int num_threads_block = 16;
        dim3 block_dim(num_threads_block, num_threads_block);
        dim3 grid_dim(
            (source->columns + num_threads_block - 1) / num_threads_block,
            (source->rows + num_threads_block - 1) / num_threads_block
        );

        copy_matrix_kernel<<<grid_dim, block_dim>>> (source_gpu, destination_gpu, source->rows, source->columns);

        cudaMemcpy(destination->data->elements, destination_gpu, matrix_size, cudaMemcpyDeviceToHost);

        cudaFree(source_gpu);
        cudaFree(destination_gpu);
    }


#ifdef __cplusplus
}
#endif
