#include "nvector.h"

Vector* create_vector(int size) {
    Vector* vector = malloc(sizeof(Vector));
    vector->elements = calloc(size, sizeof(double));
    vector->size = size;
    
    return vector;
}

void free_vector(Vector* vector) {
    free(vector->elements);
    free(vector);
}

void fill_vector_random(Vector* vector, double min, double max) {
    for(int i = 0; i < vector->size; i++) {
        vector->elements[i] = ((double)rand() / (double)RAND_MAX) * (max - min) + min;
    }
}

void fill_vector(Vector* vector, double value) {
    memset(vector->elements, value, vector->size * sizeof(double));
}

char* vector_to_string(const Vector* vector) {
    char* output = (char*) malloc(sizeof(char) * vector->size * 20); // Assume up to 20 characters per number
    if (output == NULL) {
        // handle error
        return NULL;
    }

    sprintf(output, "[");
    int i;
    for(i = 0; i < vector->size - 1; i++) {
        char buffer[20];
        sprintf(buffer, "%f, ", vector->elements[i]);
        strcat(output, buffer);
    }
    
    // Add last element and closing bracket
    char buffer[20];
    sprintf(buffer, "%f]", vector->elements[i]);
    strcat(output, buffer);

    return output;
}


Vector* copy_vector(const Vector* vector) {
    Vector* copy = create_vector(vector->size);

    for(int i = 0; i < vector->size; i++) {
        copy->elements[i] = vector->elements[i];
    }
    
    return copy;
}

Vector* slice_vector(const Vector* vector, int beginIndex, int endIndex) {
    int newSize = endIndex - beginIndex ;
    Vector* newVector = create_vector(newSize);

    for (int i = beginIndex, j = 0; i <= endIndex; i++, j++) {
        newVector->elements[j] = vector->elements[i];
    }

    return newVector;
}

Vector* array_to_vector(double* array, int length) {
    Vector* vector = create_vector(length);

    for(int i = 0; i < vector->size; i++) {
        vector->elements[i] = array[i];
    }

    return vector;
}