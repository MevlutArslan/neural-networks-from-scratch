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

#endif