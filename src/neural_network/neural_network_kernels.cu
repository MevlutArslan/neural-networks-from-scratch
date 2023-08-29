#include "nnetwork.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>

__global__ void calculate_weight_gradients_kernel(double* loss_wrt_weightedsum, double* wsum_wrt_weight, double* accumulated_gradients, int loss_wrt_weightedsum_columns, int wsum_wrt_weight_row_count, int wsum_wrt_weight_columns) {
    int row_begin = blockIdx.x * blockDim.x;
    int row_end = row_begin + blockDim.x < wsum_wrt_weight_row_count ? row_begin + blockDim.x : wsum_wrt_weight_row_count;

    int weight_gradient_row_length = loss_wrt_weightedsum_columns * wsum_wrt_weight_columns;
    int larger_accumulation_array_index = blockIdx.x * weight_gradient_row_length;

    if(row_begin < wsum_wrt_weight_row_count) {
        for(int row_index = row_begin; row_index < row_end; row_index++) {
            double scalar = 0;

            for (int i = 0; i < loss_wrt_weightedsum_columns; i++) {
                int scalar_index = (row_index * loss_wrt_weightedsum_columns) + i;
                scalar = loss_wrt_weightedsum[scalar_index];

                for (int j = 0; j < wsum_wrt_weight_columns; j++) {
                    int flat_index = (row_index * wsum_wrt_weight_columns) + j;
                    double product_result = scalar * wsum_wrt_weight[flat_index];

                    // Calculate the index in the extended accumulation array by adding the local (i, j) index
                    // within this block to the starting index of this block's subrange in the accumulation array.
                    int accumulated_gradients_flat_index = ((i * wsum_wrt_weight_columns) + j) + larger_accumulation_array_index;
                    double current_gradient = accumulated_gradients[accumulated_gradients_flat_index];
                    double new_gradient = current_gradient + product_result;

                    accumulated_gradients[accumulated_gradients_flat_index] = new_gradient;
                }
            }
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif

void calculate_weight_gradients_cuda(NNetwork* network, int layer_index, Matrix* loss_wrt_weightedsum, Matrix* wsum_wrt_weight) {
    int threads_per_block = 32;
    int num_blocks = (wsum_wrt_weight->rows + threads_per_block - 1) / threads_per_block;

    dim3 grid_size(num_blocks);
    dim3 block_size(threads_per_block);

    // Each kernel processes a subrange of rows, e.g., 1-100, 100-200, 200-300.
    // Each kernel accumulates gradients for its assigned subrange and writes to its portion
    // in the larger accumulation array. After all kernels complete their work, we sum
    // these partial results and write them to the gradient matrix.
    double* loss_wrt_wsum_device;
    double* wsum_wrt_weight_device;
    double* accumulated_values_device; 

    size_t loss_wrt_wsum_size = loss_wrt_weightedsum->rows * loss_wrt_weightedsum->columns * sizeof(double);
    size_t wsum_wrt_weight_size = wsum_wrt_weight->rows * wsum_wrt_weight->columns * sizeof(double);
    size_t accumulated_values_size = (network->weight_gradients[layer_index]->rows * network->weight_gradients[layer_index]->columns * sizeof(double)) 
                                      * num_blocks;

    cudaMalloc(&loss_wrt_wsum_device, loss_wrt_wsum_size);
    cudaMalloc(&wsum_wrt_weight_device, wsum_wrt_weight_size);
    cudaMalloc(&accumulated_values_device, accumulated_values_size);

    cudaMemcpy(loss_wrt_wsum_device, loss_wrt_weightedsum->data->elements, loss_wrt_wsum_size, cudaMemcpyHostToDevice);
    cudaMemcpy(wsum_wrt_weight_device, wsum_wrt_weight->data->elements, wsum_wrt_weight_size, cudaMemcpyHostToDevice);

    cudaMemset(&accumulated_values_device, 0.0, network->weight_gradients[layer_index]->rows * network->weight_gradients[layer_index]->columns * num_blocks);
    
    calculate_weight_gradients_kernel<<<grid_size, block_size>>>(loss_wrt_wsum_device, wsum_wrt_weight_device, accumulated_values_device, loss_wrt_weightedsum->columns, wsum_wrt_weight->rows, wsum_wrt_weight->columns);

    cudaDeviceSynchronize();

    // Perform the final reduction and copy accumulated values back to the host
    double* partial_accumulated_gradients_host = (double*)malloc(accumulated_values_size);
    cudaMemcpy(partial_accumulated_gradients_host, accumulated_values_device, accumulated_values_size, cudaMemcpyDeviceToHost);
    int weight_gradient_row_length = loss_wrt_weightedsum->columns * wsum_wrt_weight->columns;

    for (int i = 0; i < num_blocks; i++) {
        for (int j = 0; j < weight_gradient_row_length; j++) {
            int index = i * weight_gradient_row_length + j;
            network->weight_gradients[layer_index]->data->elements[j] += partial_accumulated_gradients_host[index];
        }
    }

    cudaFree(loss_wrt_wsum_device);
    cudaFree(wsum_wrt_weight_device);
    cudaFree(accumulated_values_device);

    free(partial_accumulated_gradients_host);
} 

#ifdef __cplusplus
}
#endif