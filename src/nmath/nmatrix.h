#ifndef NMATRIX_H
#define NMATRIX_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "nvector.h"

typedef struct {
    int rows;
    int columns;
    int isIdentity;
    Vector** data;
} Matrix;

Matrix* createMatrix(const int rows, const int cols);

void freeMatrix(Matrix* matrix);

void printMatrix(const Matrix* matrix);

int isEqual(const Matrix* m1, const Matrix* m2);

int isSquare(const Matrix* m);

// TODO : Rename this to whatever it is called!
Matrix* generateMiniMatrix(const Matrix* m, int excludeRow, int excludeColumn);

#endif