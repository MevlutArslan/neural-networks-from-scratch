#include "nmath.h"
#include "cuda.h"
#include "cuda_runtime.h"

/* ANALYZING
    m = a.rows
    n = a.cols
    k = c.cols
    __global__ void gpu_matrix_mult(int *a,int *b, int *c, int m, int n, int k)
    { 
        int row = blockIdx.y * blockDim.y + threadIdx.y; <- (row_index * rows width(aka num_cols in per row) + column_index)
        int col = blockIdx.x * blockDim.x + threadIdx.x; <- (col_index * col height(aka num_rows in the matrix) + row_index)
        int sum = 0;
        if( col < k && row < m)  // if our row and col indices are in range of the output matrix
        {
            for(int i = 0; i < n; i++)  // iterate over all columns in this row and sum up the values at a and b.
            {
                sum += a[row * n + i] * b[i * k + col]; 
            }
            c[row * k + col] = sum; // write the result at the output matrices index for this thread
        }
    } 
*/

__global__ void matrix_product_kernel(double* m1, double* m2, double* m3, int m1_cols, int output_cols) {
    // Calculate the global row and column indices in the output matrix
    int global_row = blockIdx.x;
    int global_col = threadIdx.x;

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

#ifdef __cplusplus
extern "C" {
#endif

void matrix_product_cuda(Matrix* m1, Matrix* m2, Matrix* output) {
    // m1.cols has to be equal m2.rows && output should be NULL
    assert(m1->columns == m2->rows);
    assert(output != NULL);

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
    dim3 grid_dims(m1->rows + threads_per_block - 1 / threads_per_block);
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
}

#ifdef __cplusplus
}
#endif
