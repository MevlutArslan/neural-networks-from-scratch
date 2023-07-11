#ifndef NMATRIX_H
#define NMATRIX_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "nvector.h"

typedef struct {
    int rows;
    int columns;
    Vector** data;
} Matrix;

Matrix* createMatrix(const int rows, const int cols);

void initializeMatrixWithRandomValuesInRange(Matrix* matrix, double min, double max);
void fillMatrix(Matrix* matrix, double value);

void freeMatrix(Matrix* matrix);

char* matrixToString(const Matrix* matrix);

int isEqual(const Matrix* m1, const Matrix* m2);

int isSquare(const Matrix* m);

Matrix* generateMiniMatrix(const Matrix* m, int excludeRow, int excludeColumn);

#endif
