#include "nmath.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
/*
    * The value of grid_size should not be lower than the number of SMs on the GPU, otherwise there will be SMs in the idle state.
    * For a general elementwise kernel, the total number of threads should be no greater than the total number of elements.


*/

/*
    Each row of m1 is assigned to a block.
    Each thread represents a column in that row.
*/
__global__ void matrix_product_kernel(double* m1, double* m2, double* m3, int m1_rows, int m1_columns, int output_cols) {
    // Calculate the global row and column indices in the output matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m1_rows && col < output_cols) {
        double sum = 0.0;
        for (int i = 0; i < m1_columns; i++) {
            // Calculate the indices in the input matrices
            int index_m1 = row * m1_columns + i;
            int index_m2 = i * output_cols + col;

            sum += m1[index_m1] * m2[index_m2];
        }

        // Store the result in the output matrix
        int output_index = row * output_cols + col;
        m3[output_index] = sum;
    }
}

__global__ void matrix_vector_addition_kernel(double* matrix, double* vector, double* output, int rows, int cols) {    
    int start_row = threadIdx.y + (blockIdx.y * blockDim.y);
    int start_col = threadIdx.x + (blockIdx.x * blockDim.x);

    int end_row = start_row + blockDim.y;
    int end_col = start_col + blockDim.x;

    if(start_row < rows && start_col < cols) {
        // Each block handles a mini matrix
        int start_index = start_row * cols + start_col;
        int end_index = start_row * cols + end_col;
        
        for(int i = start_index, col_index = start_col; i < end_index && col_index < cols; i++, col_index++) {
            output[i] = matrix[i] + vector[col_index];
        }
    }
}

// TODO: FIX INDEXING
__global__ void matrix_multiplication_kernel(double* m1, double* m2, double* output, int rows, int cols) {
    int row = (blockIdx.y * blockDim.y) + threadIdx.y;
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;

    int index = row * cols + col;

    if(index < rows * cols) {
        output[index] = m1[index] * m2[index];
    }
}

__global__ void matrix_addition_kernel(double* m1, double* m2, double* output, int rows, int cols) {
    int row = (blockIdx.y * blockDim.y) + threadIdx.y;
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;

    int index = row * cols + col;

    if(index < rows * cols) {
        output[index] = m1[index] + m2[index];
    }
}

__global__ void matrix_subtraction_kernel(double* m1, double* m2, double* output, int rows, int cols) {
    int row = (blockIdx.y * blockDim.y) + threadIdx.y;
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;

    int index = row * cols + col;

    if(index < rows * cols) {
        output[index] = m1[index] - m2[index];
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

    int threads_per_block = 16;
    // set dimensions for distributing the work
    dim3 grid_dims(
        (m2->columns + threads_per_block - 1) / threads_per_block,
        (m1->rows + threads_per_block - 1) / threads_per_block
        );
    dim3 block_dims(threads_per_block, threads_per_block);

    // run kernel
    matrix_product_kernel<<<grid_dims, block_dims>>>(m1_gpu, m2_gpu, output_gpu, m1->rows, m1->columns, output->columns);

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

void matrix_vector_addition_cuda(Matrix* matrix, Vector* vector, Matrix* output) {
    double* m_gpu;
    double* v_gpu;
    double* output_gpu;

    size_t m_size = matrix->rows * matrix->columns * sizeof(double);
    size_t v_size = vector->size * sizeof(double);
    size_t output_size = m_size;

    cudaMalloc(&m_gpu, m_size);
    cudaMalloc(&v_gpu, v_size);
    cudaMalloc(&output_gpu, output_size);

    cudaMemcpy(m_gpu, matrix->data->elements, m_size, cudaMemcpyHostToDevice);
    cudaMemcpy(v_gpu, vector->elements, v_size, cudaMemcpyHostToDevice);

    // each block is 16 threads by 16 threads.
    int num_threads_block = 16;
    dim3 block_dim(num_threads_block, num_threads_block);

    dim3 grid_dim(
        // height of the grid
        (matrix->columns + num_threads_block - 1) / num_threads_block,
        // width of the grid
        (matrix->rows + num_threads_block - 1) / num_threads_block
    );

    matrix_vector_addition_kernel<<<grid_dim, block_dim>>>(m_gpu, v_gpu, output_gpu, matrix->rows, matrix->columns);

    cudaDeviceSynchronize();
    cudaMemcpy(output->data->elements, output_gpu, output_size, cudaMemcpyDeviceToHost);

    cudaFree(m_gpu);
    cudaFree(v_gpu);
    cudaFree(output_gpu);
}

Matrix* matrix_element_wise_operation_cuda(Matrix* m1, Matrix* m2, ElementWiseMatrixOperation operation) {
    assert(m1->rows == m2->rows && m1->columns == m2->columns);

    Matrix* output = create_matrix(m1->rows, m2->columns);

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
    
    // each block is 16 threads by 16 threads.
    int num_threads_block = 16;
    dim3 block_dim(num_threads_block, num_threads_block);
    dim3 grid_dim(
        (m1->columns + num_threads_block - 1) / num_threads_block,
        (m1->rows + num_threads_block - 1) / num_threads_block
    );
    // kernel
    switch(operation) {
        case ADD:
            matrix_addition_kernel<<<grid_dim, block_dim>>> (m1_gpu, m2_gpu, output_gpu, m1->rows, m1->columns);
            break;
        case SUBTRACT:
            matrix_subtraction_kernel<<<grid_dim, block_dim>>> (m1_gpu, m2_gpu, output_gpu, m1->rows, m1->columns);
            break;
        case MULTIPLY:
            matrix_multiplication_kernel<<<grid_dim, block_dim>>> (m1_gpu, m2_gpu, output_gpu, m1->rows, m1->columns);
            break;
        default:
            break;
    }

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
