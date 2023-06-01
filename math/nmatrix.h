#ifndef NMATRIX_H
#define NMATRIX_H

#include <stdlib.h>
#include <stdio.h>

typedef struct {
    int rows;
    int columns;

    double** data;
} Matrix;


Matrix* createMatrix(const int rows, const int cols);
void freeMatrix(Matrix* matrix);

void printMatrix(const Matrix* matrix);

#endif