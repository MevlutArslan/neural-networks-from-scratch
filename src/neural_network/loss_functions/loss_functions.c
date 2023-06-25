#include "loss_functions.h"
#include <stdio.h>
double meanSquaredError(Vector* actual, Vector* predicted) {
    Vector* result = createVector(actual->size);

    double sum = 0.0f;
    for(int i = 0; i < result->size; i++) {
        result->elements[i] = actual->elements[i] - predicted->elements[i];
        result->elements[i] = result->elements[i] * result->elements[i];
        sum += result->elements[i];

    }

    deleteVector(result);
    return sum / actual->size;
}