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

void mock_forward_pass(Matrix* input_matrix, Matrix** layer_outputs, int num_layers, Matrix** weights, Vector** biases, int begin_index, int end_index);
FILE* logFile;

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
        [ ]. implement the logic from the mock forward pass into NNetwork
            [ ]. verify the resuls by comparing them to old version's calculations


        BACKWARD PASS

    */


    // two layers with [4, 2] neurons each
    int num_layers = 2;

    Matrix* input_matrix = create_matrix(2, 4);
    fill_matrix_random(input_matrix, -2, 5);

    Matrix* weights_1 = create_matrix(4, input_matrix->columns);
    fill_matrix_random(weights_1, 0.0f, 1.0f);

    Vector* biases_1 = create_vector(4);
    fill_vector_random(biases_1, -0.5f, 0.5f);

    Matrix* weights_2 = create_matrix(2, weights_1->rows);
    fill_matrix_random(weights_2, 0.0f, 1.0f);

    Vector* biases_2 = create_vector(2);
    fill_vector_random(biases_2, -0.5f, 0.5f);

    Matrix** weights[num_layers];
    weights[0] = weights_1;
    weights[1] = weights_2;

    Vector** biases[num_layers];
    biases[0] = biases_1;
    biases[1] = biases_2;
    // input rows x output size
    Matrix** layer_outputs =  create_matrix_arr(num_layers);
    Matrix** row_layer_outputs;
    log_info("Before mock forward pass!");

    log_info("Input Matrix: %s", matrix_to_string(input_matrix));
    log_info("Weights #1: %s", matrix_to_string(weights_1));
    log_info("Weights #2 %s", matrix_to_string(weights_2));

    log_info("Biases #1: %s", vector_to_string(biases_1));
    log_info("Biases #2: %s", vector_to_string(biases_2));

    mock_forward_pass(input_matrix, layer_outputs, num_layers, weights, biases, 0, input_matrix->rows);
}   


void softmax_matrix(Matrix* matrix) {
    for(int i = 0; i < matrix->rows; i++) {
        softmax(matrix->data[i]);
    }
}

void mock_forward_pass(Matrix* input_matrix, Matrix** layer_outputs, int num_layers, Matrix** weights, Vector** biases, int begin_index, int end_index) { 
    for(int i = 0; i < num_layers; i++) {
        Matrix* product;
        Matrix* transposed_weights = matrix_transpose(weights[i]);
        
        if(i == 0) {
            product = matrix_product(input_matrix, transposed_weights);
            log_info("product result for first layer: %s", matrix_to_string(product));
        }else{
            product = matrix_product(layer_outputs[i - 1], transposed_weights);
            log_info("product result for second layer: %s", matrix_to_string(product));
        }

        layer_outputs[i] = matrix_vector_addition(product, biases[i]);
        log_info("Vector addition results: %s", matrix_to_string(layer_outputs[i]));

        if(i == num_layers - 1) {
            softmax_matrix(layer_outputs[i]);
        }else {
            leakyReluMatrix(layer_outputs[i]);
        }

        log_info("After activation: %s", matrix_to_string(layer_outputs[i]));
    }
}