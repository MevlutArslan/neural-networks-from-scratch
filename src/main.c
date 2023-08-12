#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../tests/test.h"
#include <time.h>
#include "../libraries/logger/log.h"
#include "example_networks/mnist/mnist.h"
#include "example_networks/wine_dataset/wine_dataset.h"
#include "helper/thread_pool.h"
#include "nmath/nmath.h"
#include "nmath/nmatrix.h"
#include "nmath/nvector.h"
#include "nmath/math_tasks.h"

void runProgram();

#define TIME_BUFFER_SIZE 64

FILE* logFile;
void mock_forward_pass(Matrix* input_matrix, Matrix** layer_outputs, int num_layers, Matrix** weights, Vector** biases, Matrix** layer_weighedsums);
void mock_backward_pass(Matrix* input_matrix, Matrix** layer_outputs, int num_layers, Matrix* y_hats, Matrix** weight_gradients, Vector** bias_gradients, Matrix** weights, Matrix** layer_weighedsums);

void computeCategoricalCrossEntropyLossDerivativeMatrix(Matrix* target, Matrix* prediction, Matrix* loss_wrt_output);
Matrix** softmax_derivative_parallelized(Matrix* output);

Matrix** matrix_product_arr(Matrix** matrix_arr, Matrix* matrix, int size);
Matrix* matrix_vector_product_arr(Matrix** matrix_arr, Matrix* matrix, int size);

Matrix* leakyRelu_derivative_matrix(Matrix* input);

void file_log(log_Event *ev) {
    char time_buffer[TIME_BUFFER_SIZE];
    strftime(time_buffer, TIME_BUFFER_SIZE, "%Y-%m-%d %H:%M:%S", ev->time);
  
    fprintf(logFile, "%s %-5s %s:%d: ", time_buffer, log_level_string(ev->level), ev->file, ev->line);
    vfprintf(logFile, ev->fmt, ev->ap);
    fprintf(logFile, "\n");
    fflush(logFile);
}

struct ForwardPassArgs {
    Matrix* input_matrix;
    Matrix** layer_outputs;
    
    Matrix** weights;
    Vector** biases;

  
    int num_layers;
    int begin_index;
    int end_index;
};

int main(int argc, char* argv[])
{
    logFile = fopen("log.txt", "w");
    if (logFile == NULL) {
        printf("Failed to open log file.\n");
        return 0;
    }

    // Add the file_log as a callback to the logging library
    log_add_callback(file_log, NULL, LOG_TRACE);
    
    srand(306); // seeding with 306

    int isTesting = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "test") == 0) {
            isTesting = 1;
            break;
        }
    }

    if (isTesting) {
        log_info("%s", "Running tests...");
        run_tests();
    } else {
        log_info("%s", "Running Program!");
        runProgram();   
    }
}

void runProgram() {
    // Model* model = create_wine_categorization_model();
    // clock_t start = clock();

    // model->train_network(model);
    
    // clock_t end = clock();

    // log_info("time it took to process mnist: %f",  (((double) (end - start)) / CLOCKS_PER_SEC) * 1000);
    // free_model(model);

    /* 
        FORWARD PASS
        [X]. push through the 'input' matrix via a mock forward pass.
            [X]. verify the results
            [X]. dot product
            [X]. matrix-vector addition
            [X]. activation function
        [X]. try to do it without using layer's properties that is not thread safe.
            [X]. identify properties that are not thread safe
        ---------------------------------------------------------------------------
        [X]. modify the mock forward pass to work with "batches" instead of using the entire input matrix
            [X]. verify the results
        [-]. run the mini batch version using the thread pool 
            [-]. verify the results
        ---------------------------------------------------------------------------
        [X]. implement the logic from the mock forward pass into NNetwork
            [X]. verify the resuls by comparing them to old version's calculations

        BACKWARD PASS
    */

    // two layers with [4, 2] neurons each
    int num_layers = 2;

    Matrix* input_matrix = create_matrix(3, 4);
    fill_matrix_random(input_matrix, -2, 5);

    Matrix* weights_1 = create_matrix(4, input_matrix->columns);
    fill_matrix_random(weights_1, 0.0f, 1.0f);

    Vector* biases_1 = create_vector(4);
    fill_vector_random(biases_1, -0.5f, 0.5f);

    Matrix* weights_2 = create_matrix(2, weights_1->rows);
    fill_matrix_random(weights_2, 0.0f, 1.0f);

    Vector* biases_2 = create_vector(2);
    fill_vector_random(biases_2, -0.5f, 0.5f);

    Matrix** weights = create_matrix_arr(num_layers);
    weights[0] = weights_1;
    weights[1] = weights_2;

    Vector** biases = create_vector_arr(num_layers);
    biases[0] = biases_1;
    biases[1] = biases_2;
    // input rows x output size
    Matrix** layer_outputs = create_matrix_arr(num_layers);

    Matrix** layer_wsums = create_matrix_arr(num_layers);
    
    Vector* target_values = create_vector(input_matrix->rows);
    target_values->elements[0] = 1;
    target_values->elements[1] = 2;

    Matrix* y_hats = oneHotEncode(target_values, 2);

    // for each input row, for each layer in that row, weight gradients

    // Matrix*** weight_gradients = (Matrix***) malloc(num_layers * sizeof(Matrix**));
    // for(int i = 0; i < weight_gradients; i++) {
    //     weight_gradients[i] = create_matrix_arr();
    // }

    // for each layer we store its weight gradient matrix
    Matrix** weight_gradients = create_matrix_arr(num_layers);
    for(int i = 0; i < num_layers; i++) {
        weight_gradients[i] = create_matrix(weights[i]->rows, weights[i]->columns);
        fill_matrix(weight_gradients[i], 0.0f);
    }

    Vector** bias_gradients = create_vector_arr(num_layers);
    for(int i = 0; i < num_layers; i++) {
        bias_gradients[i] = create_vector(biases[i]->size);
        fill_vector(bias_gradients[i], 0.0f);
    }

    log_info("Before mock forward pass!");

    log_info("Input Matrix: %s", matrix_to_string(input_matrix));
    log_info("Weights #1: %s", matrix_to_string(weights_1));
    log_info("Weights #2 %s", matrix_to_string(weights_2));

    log_info("Biases #1: %s", vector_to_string(biases_1));
    log_info("Biases #2: %s", vector_to_string(biases_2));

    log_info("Weight gradients #1: %s", matrix_to_string(weight_gradients[0]));
    log_info("Weight gradients #2: %s", matrix_to_string(weight_gradients[1]));

    log_info("Y_Hats: %s", matrix_to_string(y_hats));

    mock_forward_pass(input_matrix, layer_outputs, num_layers, weights, biases, layer_wsums);

    log_info("output of the forward pass: %s", matrix_to_string(layer_outputs[num_layers - 1]));

    // mock backward pass
    // the output's pass
    /*
        a. categoricalCrossEntropyLossDerivative for each input
        b. softmax_derivative for each input
        c. dLoss_dWeightedSums for each input
        d. calculate gradients for weight gradients
        e. gradient clipping logic for weights
        f. calculate bias gradients
        g. gradient clipping logic for biase gradients
        h. propagate the loss_wrt_wsums of the current layer to the previous layer.
    */
   
    mock_backward_pass(input_matrix, layer_outputs, num_layers, y_hats, weight_gradients, bias_gradients, weights, layer_wsums);    
}   

void mock_backward_pass(Matrix* input_matrix, Matrix** layer_outputs, int num_layers, Matrix* y_hats, Matrix** weight_gradients, Vector** bias_gradients, Matrix** weights, Matrix** layer_weightedsums) {
    // -------------OUTPUT LAYER-------------
    int layer_index = num_layers - 1;
    Matrix* output = layer_outputs[num_layers - 1];

    // I can distribute the work amongst the threads in the thread pool for all three operations.
    Matrix* loss_wrt_output = create_matrix(y_hats->rows, y_hats->columns);
    computeCategoricalCrossEntropyLossDerivativeMatrix(y_hats, output, loss_wrt_output);

    log_info("loss_wrt_output: %s", matrix_to_string(loss_wrt_output));

    Matrix** jacobian_matrices = softmax_derivative_parallelized(output);

    // not very sure about this.
    Matrix** loss_wrt_weightedsum = create_matrix_arr(num_layers);
    loss_wrt_weightedsum[layer_index] = matrix_vector_product_arr(jacobian_matrices, loss_wrt_output, output->rows);
    
    log_info("Loss wrt WeightedSum matrix for layer #%d: %s", layer_index, matrix_to_string(loss_wrt_weightedsum[layer_index]));

    Matrix* weightedsum_wrt_weight;    

    if (layer_index == 0) {
        weightedsum_wrt_weight = input_matrix;
    } else {
        weightedsum_wrt_weight = layer_outputs[layer_index - 1];
    }
    log_info("weightedsum wrt weights for layer #%d: %s", layer_index, matrix_to_string(weightedsum_wrt_weight));

    // multiplying each weighted sum with different input neurons to get the gradients of the weights that connect them
    for(int input_index = 0; input_index < input_matrix->rows; input_index++) {
        for(int i = 0; i < loss_wrt_weightedsum[layer_index]->columns; i++) {
            for(int j = 0; j < input_matrix->columns; j++) {
                weight_gradients[layer_index]->data[i]->elements[j] += loss_wrt_weightedsum[layer_index]->data[input_index]->elements[i] * weightedsum_wrt_weight->data[input_index]->elements[j];
            }
        }
    }
    
    log_info("Weight gradients of the output layer: %s", matrix_to_string(weight_gradients[layer_index]));

    // if(useGradientClipping) clip_gradients(weight_gradients)

    for(int i = 0; i < loss_wrt_weightedsum[layer_index]->rows; i++) {
        for(int j = 0; j < loss_wrt_weightedsum[layer_index]->columns; j++) {
            bias_gradients[layer_index]->elements[j] += loss_wrt_weightedsum[layer_index]->data[i]->elements[j];
        }
    }
    
    log_info("Bias gradients for the output layer: %s", vector_to_string(bias_gradients[layer_index]));
    // if(useGradientClipping) clip_gradients(bias_gradients)

    // ------------- HIDDEN LAYERS -------------
    // we do need to iterate over other layers
    for (layer_index -= 1; layer_index >= 0; layer_index--) { // current layer's dimensions = (4 inputs, 4 neurons)
        /*
            a. we calculate loss_wrt_output for each neuron in the current layer by adding up (next layer's loss_wrt_wsum * next layer's weights)
            b. we calculate output_wrt_wsum by using the derivative of the activation function of the current layer for the weighted sum
            c. we calcilate loss_wrt_wsum by multiplying loss_wrt_output & output_wrt_wsum
            d. we calculate weightedsum_wrt_weight
            e. we calculate weight gradients of the layer
            f. gradient clipping logic for weight gradients
            g. we calculate bias gradients of the layer
            h. gradient clipping logic for bias gradients
            i. propagate the loss_wrt_wsums of the current layer to the previous layer.
        */

        // NOT SURE, DOUBLE CHECK IF THE RESULTS ARE NOT MATCHING
        // for each input row, we store loss_wrt_output of neurons in the columns
        // each column in each row will be summation of all of the next layer's loss_wrt_weighted sum values and next layer's weights
        Matrix* loss_wrt_output = matrix_product(loss_wrt_weightedsum[layer_index + 1], weights[layer_index+1]);
        log_info("loss wrt output for layer: #%d: %s", layer_index, matrix_to_string(loss_wrt_output));

        Matrix* output_wrt_weightedsums = leakyRelu_derivative_matrix(layer_weightedsums[layer_index]);
        log_info("output wrt wsum for layer #%d: %s", layer_index, matrix_to_string(output_wrt_weightedsums));

        loss_wrt_weightedsum[layer_index] = matrix_multiplication(loss_wrt_output, output_wrt_weightedsums);
        log_info("loss wrt weighted sum for layer #%d: %s", layer_index, matrix_to_string(loss_wrt_weightedsum[layer_index]));

        if (layer_index == 0) {
            weightedsum_wrt_weight = input_matrix;
        } else {
            weightedsum_wrt_weight = layer_outputs[layer_index - 1];
        }

        log_info("weightedsum wrt weights for layer #%d: %s", layer_index, matrix_to_string(weightedsum_wrt_weight));

        // NOT SURE
        // multiplying each weighted sum with different input neurons to get the gradients of the weights that connect them
        for(int input_index = 0; input_index < input_matrix->rows; input_index++) {
            for(int i = 0; i < loss_wrt_weightedsum[layer_index]->columns; i++) {
                for(int j = 0; j < input_matrix->columns; j++) {
                    weight_gradients[layer_index]->data[i]->elements[j] += loss_wrt_weightedsum[layer_index]->data[input_index]->elements[i] * weightedsum_wrt_weight->data[input_index]->elements[j];
                }
            }
        }
        
        log_info("Weight gradients of the layer #%d: %s", layer_index, matrix_to_string(weight_gradients[layer_index]));

        // if(useGradientClipping) clip_gradients(weight_gradients)

        for(int i = 0; i < loss_wrt_weightedsum[layer_index]->rows; i++) {
            for(int j = 0; j < loss_wrt_weightedsum[layer_index]->columns; j++) {
                bias_gradients[layer_index]->elements[j] += loss_wrt_weightedsum[layer_index]->data[i]->elements[j];
            }
        }
    
        log_info("Bias gradients of the layer #%d: %s", layer_index, vector_to_string(bias_gradients[layer_index]));
    }
}

Matrix* leakyRelu_derivative_matrix(Matrix* input) {
    Matrix* result = create_matrix(input->rows, input->columns);

    for(int i = 0; i < result->rows; i++) {
        for(int j = 0; j < result->columns; j++) {
            result->data[i]->elements[j] = leakyRelu_derivative(input->data[i]->elements[j]);
        }
    }

    return result;
}

Matrix** matrix_product_arr(Matrix** matrix_arr, Matrix* matrix, int size) {
    Matrix** result_arr = create_matrix_arr(size);

    for(int i = 0; i < size; i++) {
        result_arr[i] = matrix_product(matrix_arr[i], matrix);
    }

    return result_arr;
}

Matrix* matrix_vector_product_arr(Matrix** matrix_arr, Matrix* matrix, int size) {
    Matrix* result = create_matrix(matrix->rows, matrix_arr[0]->columns);

    for(int i = 0; i < matrix->rows; i++) {
        result->data[i] = dot_product(matrix_arr[i], matrix->data[i]);
    }

    return result;
}

Matrix** softmax_derivative_parallelized(Matrix* output) {
    Matrix** jacobian_matrices = create_matrix_arr(output->rows);
    for(int i = 0; i < output->rows; i++) {
        jacobian_matrices[i] = softmax_derivative(output->data[i]);
        log_info("Jacobian matrix #%d: %s", i, matrix_to_string(jacobian_matrices[i]));
    }

    return jacobian_matrices;
}

void computeCategoricalCrossEntropyLossDerivativeMatrix(Matrix* target, Matrix* prediction, Matrix* loss_wrt_output) {
    for(int i = 0; i < target->rows; i++) {
        loss_wrt_output->data[i] = categoricalCrossEntropyLossDerivative(target->data[i], prediction->data[i]);
    }
}

void mock_forward_pass(Matrix* input_matrix, Matrix** layer_outputs, int num_layers, Matrix** weights, Vector** biases, Matrix** layer_weightedsums) { 
    for(int i = 0; i < num_layers; i++) {
        Matrix* transposed_weights = matrix_transpose(weights[i]);
        
        if(i == 0) {
            layer_weightedsums[i] = matrix_product(input_matrix, transposed_weights);
            // log_info("product result for first layer: %s", matrix_to_string(layer_weightedsums[i]));
        }else{
            layer_weightedsums[i] = matrix_product(layer_outputs[i - 1], transposed_weights);
            // log_info("product result for second layer: %s", matrix_to_string(layer_weightedsums[i]));
        }
        

        layer_outputs[i] = matrix_vector_addition(layer_weightedsums[i], biases[i]);
        // log_info("Vector addition results: %s", matrix_to_string(layer_outputs[i]));

        if(i == num_layers - 1) {
            softmax_matrix(layer_outputs[i]);
        }else {
            leakyReluMatrix(layer_outputs[i]);
        }

        // log_info("After activation: %s", matrix_to_string(layer_outputs[i]));
    }
}