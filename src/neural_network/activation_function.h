#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <math.h>
#include "../nmath/nmath.h"

typedef struct {
    void (*activation)(Vector*);       // Pointer to the activation function
    double (*derivative)(double);       // Pointer to the derivative of the activation function
    const char* name;
} ActivationFunction;

extern ActivationFunction SOFTMAX;
extern ActivationFunction LEAKY_RELU;
extern ActivationFunction RELU;

void relu(Vector* inputs);
double relu_derivative(double netInput);

void leakyRelu(Vector* inputs);
double leakyRelu_derivative(double netInput);

void sigmoid(Vector* inputs);
double sigmoid_derivative(double netInput);

void softmax(Vector* inputs);
Matrix* softmax_derivative(Vector* output);

const char* get_activation_function_name(const ActivationFunction* activationFunction);
ActivationFunction get_activation_function_by_name(char* name);
#endif