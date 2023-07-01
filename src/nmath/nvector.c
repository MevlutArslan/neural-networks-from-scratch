#include "nvector.h"


Vector* createVector(int size) {
    Vector* vector = malloc(sizeof(Vector));
    vector->elements = malloc(size * sizeof(double));
    vector->size = size;
    
    return vector;
}

void freeVector(Vector* vector) {
    free(vector->elements);
    free(vector);
}

void initializeVectorWithRandomValuesInRange(Vector* vector, double min, double max) {
    for(int i = 0; i < vector->size; i++) {
        vector->elements[i] = ((double)rand() / (double)RAND_MAX) * (max - min) + min;
    }
}

void fillVector(Vector* vector, double value) {
    memset(vector->elements, value, vector->size * sizeof(double));
}


void printVector(const Vector* vector) {
    printf("[");
    int i;
    for(i = 0; i < vector->size - 1; i++) {
        printf("%f, ", vector->elements[i]);
    }

    printf("%f", vector->elements[i]);
    printf("]\n");
}

Vector* copyVector(const Vector* vector) {
    Vector* copy = createVector(vector->size);

    for(int i = 0; i < vector->size; i++) {
        copy->elements[i] = vector->elements[i];
    }
    
    return copy;
}
