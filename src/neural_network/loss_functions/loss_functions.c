#include "loss_functions.h"
#include <stdio.h>

double meanSquaredError(Vector* actual, Vector* predicted) {
    int size = predicted->size;

    Vector* result = createVector(size);
    
    double sum = 0.0f;
    for(int i = 0; i < size; i++) {
        result->elements[i] = actual->elements[i] - predicted->elements[i];
        result->elements[i] = result->elements[i] * result->elements[i];
        sum = sum + result->elements[i];
    }

    deleteVector(result);
    return sum / (double)size;
}