#ifndef NVECTOR_H
#define NVECTOR_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../../libraries/logger/log.h"
#include "../../libraries/cJson/cJSON.h"
#include <assert.h>
#include <math.h>

typedef struct {
    double* elements;
    int size;
} Vector;

typedef struct VectorArray{
    Vector** vectors;
    int length;
} VectorArray;

Vector* create_vector(int size);
VectorArray* create_vector_arr(int length);
VectorArray* create_vector_array_with_fixed_length(int array_length, int vector_length);

void free_vector(Vector* vector);
void free_vector_arr(VectorArray* array);

void fill_vector_random(Vector* vector, double min, double max);
void fill_vector(Vector* vector, double value);

char* vector_to_string(const Vector* vector);
Vector* copy_vector(const Vector* vector);

int is_equal_vector(Vector* v1, Vector* v2);

Vector* slice_vector(const Vector* vector, int beginIndex, int endIndex);
Vector* array_to_vector(double* array, int length);

char* serialize_vector(const Vector* vector);
Vector* deserialize_vector(cJSON* json);



#endif