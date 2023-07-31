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

void file_log(log_Event *ev) {
    char time_buffer[TIME_BUFFER_SIZE];
    strftime(time_buffer, TIME_BUFFER_SIZE, "%Y-%m-%d %H:%M:%S", ev->time);
  
    fprintf(logFile, "%s %-5s %s:%d: ", time_buffer, log_level_string(ev->level), ev->file, ev->line);
    vfprintf(logFile, ev->fmt, ev->ap);
    fprintf(logFile, "\n");
    fflush(logFile);
}

int main(int argc, char* argv[])
{
    logFile = fopen("log.txt", "w");
    if (logFile == NULL) {
        printf("Failed to open log file.\n");
        return 0;
    }

    // Add the file_log as a callback to the logging library
    log_add_callback(file_log, NULL, LOG_TRACE);
    
    srand(307); // seeding with 306

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
    // Model* model = create_mnist_model();
    // clock_t start = clock();

    // model->train_network(model);
    
    // clock_t end = clock();

    // log_info("time it took to process mnist: %f",  (((double) (end - start)) / CLOCKS_PER_SEC) * 1000);
    // free_model(model);

    struct ThreadPool* thread_pool = create_thread_pool(4);
    
    Matrix* matrix = create_matrix(2000, 750);
    fill_matrix_random(matrix, -45.0f, 45.0f);

    // log_debug("matrix: %s", matrix_to_string(matrix));
    Vector* vector = create_vector(750);
    fill_vector_random(vector, -276.0f, 213.0f);
    
    Vector* output = create_vector(matrix->rows);

    int begin_index = 0;
    int end_index = matrix->rows;

    struct MatrixVectorOperation* data = create_matrix_vector_operation(matrix, vector, output, begin_index, end_index);
    
    push_dot_product_as_task(thread_pool, data);

    wait_for_all_tasks(thread_pool);
    
    log_info("output vector after tasks are done: %s", vector_to_string(output));
    log_info("Matrix rows: %d ?= output size: %d", matrix->rows, output->size);
    destroy_thread_pool(thread_pool);
}   