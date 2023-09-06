#ifndef CONSTANTS_H
#define CONSTANTS_H

enum OPTIMIZER {
    SGD,
    ADAGRAD,
    RMS_PROP, 
    ADAM
};

const char* get_optimizer_name(enum OPTIMIZER optimizer);

// DEBUG VARIABLES
// #define DEBUG

#define TRUE 1
#define FALSE 0

#endif