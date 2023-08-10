#include "activation_function.h"
#include <stdio.h>
#include <math.h>

ActivationFunction RELU = { .activation = relu, .derivative = relu_derivative, .name = "RELU" };
ActivationFunction LEAKY_RELU = { .activation = leakyRelu, .derivative = leakyRelu_derivative, .name = "LEAKY_RELU" };
ActivationFunction SOFTMAX = { .activation = softmax, .name = "SOFTMAX" };

// in place modification
void relu(Vector* vector) {
    for (int i = 0; i < vector->size; i++) {
        vector->elements[i] = fmax(0, vector->elements[i]);
    }
}

double relu_derivative(double netInput){
    return netInput > 0 ? 1.0 : 0;
}

void leakyReluMatrix(Matrix* matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        leakyRelu(matrix->data[i]);
    }
}

void leakyRelu(Vector* vector) {
    for (int i = 0; i < vector->size; i++) {
        if (vector->elements[i] < 0) {
            vector->elements[i] *= 0.01;
        }
    }
}

double leakyRelu_derivative(double netInput) {
    return netInput > 0 ? 1.0 : 0.01;
}

void sigmoid(Vector* inputs) {
    for(int i = 0; i < inputs->size; i++) {
        inputs->elements[i] = 1 / (1 + exp(-1 * inputs->elements[i]));
    }
}

double sigmoid_derivative(double netInput) {
    double sigmoid_x = 1/(1+exp(-1 * netInput));
    return sigmoid_x * (1 - sigmoid_x);
}

/* output[i] = e ^ weightedSums[i]
               --------------------
               sum of all e ^ weightedSums[i -> number of neurons in the output layer]
                                
*/  
void softmax(Vector* inputs) {
    int size = inputs->size;

    double max_value = inputs->elements[0];
    for (int i = 1; i < size; i++) {
        max_value = fmax(max_value, inputs->elements[i]);
    }
    double sum = 0.0;
    Vector* exponentialValues = create_vector(inputs->size);
    for (int i = 0; i < size; i++) {
        exponentialValues->elements[i] = exp(inputs->elements[i] - max_value);
        sum += exponentialValues->elements[i];
    }

    for (int i = 0; i < size; i++) {
        inputs->elements[i] = exponentialValues->elements[i] / sum;
    }

    free_vector(exponentialValues);
}

void softmax_matrix(Matrix* matrix) {
    for(int i = 0; i < matrix->rows; i++) {
        softmax(matrix->data[i]);
    }
}

/*
    Derivative of each output of softmax in respect to each weighted sum of output nodes.

    Quotient Rule states d_(x/y) = (d_x * y) - (x * d_y)
                                   ---------------------
                                           y^2
    
    So d_softmaxOutput[i]   (d_softmaxOutput[i] * weightedSum[i]) - (softmaxOutput[i] * d_weightedSum[i])
      ------------------- = -------------------------------------------------------------------------------
       d_weightedSum[i]                            weightedSum[i] ^ 2
*/
Matrix* softmax_derivative(Vector* output) {
    Matrix* jacobian = create_matrix(output->size, output->size);
    
    for(int i = 0; i < output->size; i++) {
        for(int j = 0; j < output->size; j++) {
            if(i == j) {
                jacobian->data[i]->elements[j] = output->elements[i] * (1 - output->elements[i]);
            }else{
                jacobian->data[i]->elements[j] = -1 * output->elements[i] * output->elements[j];
            }
        }
    }

    return jacobian;
}

const char* get_activation_function_name(const ActivationFunction* activationFunction) {
    return activationFunction->name;
}

ActivationFunction get_activation_function_by_name(char* name) {
    if(strcmp(name, LEAKY_RELU.name) == 0) {
        return LEAKY_RELU;
    }
    if(strcmp(name, SOFTMAX.name) == 0) {
        return SOFTMAX;
    }
    
    // I will let RELU be the default for now
    return RELU;
}