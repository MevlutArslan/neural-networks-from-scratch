#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "../tests/test.h"
#include "../libraries/logger/log.h"
#include "helper/thread_pool.h"
#include "nmath/nmath.h"
#include "networks/model.h"

void runProgram();

#define TIME_BUFFER_SIZE 64

int main(int argc, char* argv[])
{
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
    srand(306);

    Model* model = create_real_estate_model();
    clock_t start = clock();

    train_model(model, TRUE);
    validate_model(model);
    
    clock_t end = clock();

    log_info("time it took to process mnist: %f",  (((double) (end - start)) / CLOCKS_PER_SEC) * 1000);
    free_model(model);
}   
