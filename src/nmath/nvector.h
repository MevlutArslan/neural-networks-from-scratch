#ifndef NVECTOR_H
#define NVECTOR_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../../libraries/logger/log.h"
#include "../../libraries/cJson/cJSON.h"

typedef struct {
    double* elements;
    int size;
} Vector;

Vector* create_vector(int size);
Vector** create_vector_arr(int length);

void free_vector(Vector* vector);

void fill_vector_random(Vector* vector, double min, double max);
void fill_vector(Vector* vector, double value);

char* vector_to_string(const Vector* vector);
Vector* copy_vector(const Vector* vector);

/* Begin Index is inclusive whereas the End Index is exclusive.*/
Vector* slice_vector(const Vector* vector, int beginIndex, int endIndex);
Vector* array_to_vector(double* array, int length);

char* serialize_vector(const Vector* vector);
Vector* deserialize_vector(cJSON* json);

#endif