#include "activation_function.h"
#include <stdio.h>
#include <math.h>
// in place modification
void relu(Vector* vector) {
    for (int i = 0; i < vector->size; i++) {
        vector->elements[i] = fmax(0, vector->elements[i]);
    }
}

void leakyRelu(Vector* vector) {
    for (int i = 0; i < vector->size; i++) {
        vector->elements[i] = fmax(0.01 * vector->elements[i], vector->elements[i]);
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

