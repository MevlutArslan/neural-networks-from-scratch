#include <stdlib.h>
#include <stdio.h>
#include "nmath/nmath.h"
#include "neural_network/layer.h"
#include "neural_network/nnetwork.h"
#include <string.h>
#include "../tests/test.h"
#include <time.h>
#include "../libraries/logger/log.h"
#include "nmath/nvector.h"
#include "example_networks/mnist.h"

void runProgram();

FILE* logFile;

NNetwork* create_custom_network();

void file_log(log_Event *ev) {
  fprintf(
    logFile, "%s %-5s %s:%d: ",
    ev->time, log_level_string(ev->level), ev->file, ev->line);
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
    
    srand(time(NULL));

    int isTesting = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "test") == 0) {
            isTesting = 1;
            break;
        }
    }

    if (isTesting) {
        log_info("Running tests...");
        run_tests();
    } else {
        log_info("Running Program!");
        runProgram();   
    }
}

void runProgram() {
    train_network();
    
    fclose(logFile);
}   