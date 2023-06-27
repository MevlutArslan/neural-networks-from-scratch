#ifndef NVECTOR_H
#define NVECTOR_H

#include <stdlib.h>
#include <stdio.h>
typedef struct {
    double* elements;
    int size;
} Vector;

Vector* createVector(int size);
void deleteVector(Vector* vector);

void printVector(Vector* vector);
#endif