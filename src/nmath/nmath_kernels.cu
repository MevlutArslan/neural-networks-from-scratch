#include "nmath.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <__clang_cuda_builtin_vars.h>

/*
    Each row of m1 is assigned to a block.
    Each thread represents a column in that row.

*/
__global__ void matrix_product_kernel(double* m1, double* m2, double* m3, int m1_cols, int output_cols) {
    // Calculate the global row and column indices in the output matrix
    int global_row = blockIdx.x;
    int global_col = threadIdx.x;

    if(global_row < blockDim.x && global_col < output_cols) {
        double sum = 0.0;
        for (int i = 0; i < m1_cols; i++) {
            // Calculate the indices in the input matrices
            int index_m1 = global_row * m1_cols + i;
            int index_m2 = i * output_cols + global_col;

            sum += m1[index_m1] * m2[index_m2];
        }

        // Store the result in the output matrix
        int output_index = global_row * output_cols + global_col;
        m3[output_index] = sum;
    }
}

#ifdef __cplusplus
extern "C" {
#endif

Matrix* matrix_product_cuda(Matrix* m1, Matrix* m2) {
    // m1.cols has to be equal m2.rows && output should be NULL
    assert(m1->columns == m2->rows);
    
    Matrix* output = create_matrix(m1->rows, m2->columns);

    // allocate on gpu
    double* m1_gpu;
    double* m2_gpu;
    double* output_gpu;

    size_t m1_size = m1->rows * m1->columns * sizeof(double);
    size_t m2_size = m2->rows * m2->columns * sizeof(double);
    size_t output_size = output->rows * output->columns * sizeof(double);
    
    cudaMalloc(&m1_gpu, m1_size);
    cudaMalloc(&m2_gpu, m2_size);
    cudaMalloc(&output_gpu, output_size);

    // copy to gpu
    cudaMemcpy(m1_gpu, m1->data->elements, m1_size, cudaMemcpyHostToDevice);
    cudaMemcpy(m2_gpu, m2->data->elements, m2_size, cudaMemcpyHostToDevice);
    // i dont need to copy the output as it is empty and will be overriden.

    int threads_per_block = m2->columns;
    // set dimensions for distributing the work
    dim3 grid_dims((m1->rows + threads_per_block - 1) / threads_per_block);
    dim3 block_dims(threads_per_block);
    
    // run kernel
    matrix_product_kernel<<<grid_dims, block_dims>>>(m1_gpu, m2_gpu, output_gpu, m1->columns, output->columns);

    cudaError_t return_code = cudaDeviceSynchronize();
    if(return_code != 0) {
        printf("error returned from device: %s \n", cudaGetErrorString(return_code));
    }

    // copy back to cpu
    // i only need to copy back the output as m1 and m2 haven't been modified
    cudaMemcpy(output->data->elements, output_gpu, output_size, cudaMemcpyDeviceToHost);

    cudaFree(m1_gpu);
    cudaFree(m2_gpu);
    cudaFree(output_gpu);

    return output;
}

#ifdef __cplusplus
}
#endif
