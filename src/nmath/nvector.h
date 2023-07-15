#ifndef NVECTOR_H
#define NVECTOR_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../../libraries/logger/log.h"

typedef struct {
    double* elements;
    int size;
} Vector;

Vector* create_vector(int size);
void free_vector(Vector* vector);

void fill_vector_random(Vector* vector, double min, double max);
void fill_vector(Vector* vector, double value);

char* vector_to_string(const Vector* vector);
Vector* copy_vector(const Vector* vector);

Vector* slice_vector(const Vector* vector, int beginIndex, int endIndex);
Vector* array_to_vector(double* array, int length);

#endif