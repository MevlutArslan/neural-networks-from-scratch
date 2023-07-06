#include "activation_function.h"
#include <stdio.h>
#include <math.h>
// in place modification
void relu(Vector* vector) {
    for (int i = 0; i < vector->size; i++) {
        vector->elements[i] = fmax(0, vector->elements[i]);
    }
}

double relu_derivative(double netInput){
    return netInput > 0 ? 1.0 : 0;
}

void leakyRelu(Vector* vector) {
    for (int i = 0; i < vector->size; i++) {
        vector->elements[i] = fmax(0.01 * vector->elements[i], vector->elements[i]);
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
