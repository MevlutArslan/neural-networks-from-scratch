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
    Model* model = create_wine_categorization_model();
    clock_t start = clock();

    model->train_network(model);
    
    clock_t end = clock();

    log_info("time it took to process mnist: %f",  (((double) (end - start)) / CLOCKS_PER_SEC) * 1000);
    free_model(model);

}   
