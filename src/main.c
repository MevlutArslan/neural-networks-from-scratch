#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../tests/test.h"
#include <time.h>
#include "../libraries/logger/log.h"
#include "example_networks/mnist/mnist.h"
#include "example_networks/wine_dataset/wine_dataset.h"

void runProgram();

// #define TIME_BUFFER_SIZE 64
// void file_log(log_Event *ev) {
//     char time_buffer[TIME_BUFFER_SIZE];
//     strftime(time_buffer, TIME_BUFFER_SIZE, "%Y-%m-%d %H:%M:%S", ev->time);
  
//     fprintf(logFile, "%s %-5s %s:%d: ", time_buffer, log_level_string(ev->level), ev->file, ev->line);
//     vfprintf(logFile, ev->fmt, ev->ap);
//     fprintf(logFile, "\n");
//     fflush(logFile);
// }

int main(int argc, char* argv[])
{
    // logFile = fopen("log.txt", "w");
    // if (logFile == NULL) {
    //     printf("Failed to open log file.\n");
    //     return 0;
    // }

    // // Add the file_log as a callback to the logging library
    // log_add_callback(file_log, NULL, LOG_TRACE);
    
    srand(time(NULL));

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

    model->train_network(model);
    
    free_model(model);
}   