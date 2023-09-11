#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <math.h>
#include "../nmath/nmath.h"

#define RELU_STR "relu"
#define LEAKY_RELU_STR "leaky_relu"
#define SOFTMAX_STR "softmax"

typedef enum ActivationFunction {
    RELU, LEAKY_RELU, SOFTMAX, UNRECOGNIZED_AFN
} ActivationFunction;

void relu(Vector* weighted_sums);
void relu_batched(Matrix* weighted_sums);

double relu_derivative(double weighted_sum);
Matrix* relu_derivative_batched(Matrix* weighted_sums);

void leaky_relu(Vector* weighted_sums);
void leaky_relu_batched(Matrix* weighted_sums);

double leaky_relu_derivative(double netInput);
Matrix* leaky_relu_derivative_batched(Matrix* input);

void sigmoid(Vector* inputs);
double sigmoid_derivative(double netInput);

void softmax(Vector* inputs);
void softmax_batched(Matrix* matrix);

Matrix* softmax_derivative(Vector* output);
Matrix** softmax_derivative_batched(Matrix* output);

char* get_activation_function_name(const ActivationFunction activationFunction);
ActivationFunction get_activation_function_by_name(char* name);
#endif