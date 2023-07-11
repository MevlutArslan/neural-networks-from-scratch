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

char* vectorToString(const Vector* vector) {
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


Vector* copyVector(const Vector* vector) {
    Vector* copy = createVector(vector->size);

    for(int i = 0; i < vector->size; i++) {
        copy->elements[i] = vector->elements[i];
    }
    
    return copy;
}

Vector* spliceVector(const Vector* vector, int beginIndex, int endIndex) {
    int newSize = endIndex - beginIndex ;
    Vector* newVector = createVector(newSize);

    for (int i = beginIndex, j = 0; i <= endIndex; i++, j++) {
        newVector->elements[j] = vector->elements[i];
    }

    return newVector;
}
