#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <math.h>
#include "../nmath/nmath.h"
#include "../helper/thread_pool.h"

#define RELU_STR "relu"
#define LEAKY_RELU_STR "leaky_relu"
#define SOFTMAX_STR "softmax"

typedef enum ActivationFunction {
    RELU, LEAKY_RELU, SOFTMAX, UNRECOGNIZED_AFN
} ActivationFunction;

void relu(Vector* inputs);
double relu_derivative(double netInput);

void leakyReluMatrix(Matrix* matrix);
void leakyRelu(Vector* inputs);
double leakyRelu_derivative(double netInput);
Matrix* leakyRelu_derivative_matrix(Matrix* input);

void sigmoid(Vector* inputs);
double sigmoid_derivative(double netInput);

void softmax(Vector* inputs);
void softmax_matrix(Matrix* matrix, struct ThreadPool* thread_pool);
Matrix* softmax_derivative(Vector* output);
Matrix** softmax_derivative_parallelized(Matrix* output);

const char* get_activation_function_name(const ActivationFunction activationFunction);
ActivationFunction get_activation_function_by_name(char* name);
#endif