#include "nvector.h"

Vector* createVector(int size) {
    Vector* vector = malloc(sizeof(Vector));
    vector->elements = malloc(size * sizeof(double));
    vector->size = size;
    return vector;
}

// Function to delete a vector
void deleteVector(Vector* vector) {
    free(vector->elements);
    free(vector);
}

void printVector(Vector* vector) {
    printf("[");
    for(int i = 0; i < vector->size; i++) {
        printf("%f", vector->elements[i]);
    }
    printf("]\n");
}