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

int main(int argc, char* argv[])
{
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
    Model* model = create_wine_categorization_model();
    clock_t start = clock();

    model->train_network(model);

    model->validate_network(model);
    
    clock_t end = clock();

    log_info("time it took to process mnist: %f",  (((double) (end - start)) / CLOCKS_PER_SEC) * 1000);
    free_model(model);
}   
