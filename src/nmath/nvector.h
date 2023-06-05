#ifndef NVECTOR_H
#define NVECTOR_H

typedef struct {
    double* elements;
    int size;
} Vector;

Vector* createVector(int size);
void deleteVector(Vector* vector);

#endif