#ifndef NMATRIX_H
#define NMATRIX_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct {
    int rows;
    int columns;
    /**
     * Indicates if the matrix is an identity matrix to
     * make operating on the matrix more efficient.
    */
    int isIdentity;
    double** data;
} Matrix;

Matrix* createMatrix(const int rows, const int cols);

void freeMatrix(Matrix* matrix);

void printMatrix(const Matrix* matrix);

int isEqual(const Matrix* m1, const Matrix* m2);

int isSquare(const Matrix* m);

// TODO : Rename this to whatever it is called!
Matrix* generateMiniMatrix(const Matrix* m, int excludeRow, int excludeColumn);

#endif