#include "activation_function.h"
#include <stdio.h>

double relu(double input) {
    if(input > 0) {
        return input;
    }

    return 0;
}

void applyReLU(Vector* vector) {
    for (int i = 0; i < vector->size; i++) {
        vector->elements[i] = relu(vector->elements[i]);
    }
}

double sigmoid(double input) {
    return 1 / (1 + exp(-1 * input));
}

void softmax(Vector* input) {
    // Calculate the sum of e^x for each x in the input vector
    double sum = 0.0;
    for (int i = 0; i < input->size; i++) {
        sum += exp(input->elements[i]);
    }

    // Apply the softmax function to each element in the input vector
    for (int i = 0; i < input->size; i++) {
        input->elements[i] = exp(input->elements[i]) / sum;
    }
}

