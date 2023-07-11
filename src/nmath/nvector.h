#ifndef NVECTOR_H
#define NVECTOR_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    double* elements;
    int size;
} Vector;

Vector* createVector(int size);
void freeVector(Vector* vector);

void initializeVectorWithRandomValuesInRange(Vector* vector, double min, double max);
void fillVector(Vector* vector, double value);

char* vectorToString(const Vector* vector);
Vector* copyVector(const Vector* vector);

Vector* spliceVector(const Vector* vector, int beginIndex, int endIndex);

#endif