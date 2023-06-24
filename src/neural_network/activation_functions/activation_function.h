#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <math.h>
#include "../../nmath/nvector.h"

typedef struct {
    double (*activation)(double);       // Pointer to the activation function
    double (*derivative)(double);       // Pointer to the derivative of the activation function
} ActivationFunction;

typedef struct {
    void (*activation)(Vector*);     // Pointer to the activation function
    void (*derivative)(Vector*, Vector*);     // Pointer to the derivative of the activation function
} OutputActivationFunction;


double relu(double input);
void applyReLU(Vector* vector);
double sigmoid(double input);
void softmax(Vector* input);


#endif