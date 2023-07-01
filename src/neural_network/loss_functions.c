#include "loss_functions.h"
#include <stdio.h>

double meanSquaredError(double output, double target) {
    return ((output - target)*(output - target)) / 2;
}

double derivativeMeanSquaredError(double output, double target) {
    return -2 * (output - target);
}

double calculateAverageLoss(Vector* actual, Vector* predicted) {
    int size = predicted->size;

    Vector* result = createVector(size);
    
    double sum = 0.0f;
    for(int i = 0; i < size; i++) {
        result->elements[i] = meanSquaredError(actual->elements[i], predicted->elements[i]);
        sum = sum + result->elements[i];
    }

    freeVector(result);
    return sum / (double)size;
}