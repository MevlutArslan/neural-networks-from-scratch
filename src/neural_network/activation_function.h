#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <math.h>
#include "../nmath/nvector.h"

typedef struct {
    void (*activation)(Vector*);       // Pointer to the activation function
    double (*derivative)(double);       // Pointer to the derivative of the activation function
} ActivationFunction;

void relu(Vector* inputs);
double relu_derivative(double netInput);

void leakyRelu(Vector* inputs);
double leakyRelu_derivative(double netInput);

void sigmoid(Vector* inputs);
double sigmoid_derivative(double netInput);

#endif